from IPython.display import Image
Image('../input/pyboard-neuroshield-neurobrick/PyBoard_NeuroShield_NeuroBrick.png')
from timeit import default_timer
from collections import Counter
import math
import json
import pandas as pd
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

# The following module is the desktop/laptop side of a two-part serial communication protocol,
# with one part running on a desktop/laptop and the other running on a PyBoard connected by
# USB/serial. Its import is commented out here, since the Kaggle environment has no knowledge
# of it, and even if it did, there is no PyBoard physically connected to which to connect.

# Commented out since this demonstrative code cannot actually be run on Kaggle
# from PyBoard_NeuroShield_serial_port_interface import *
%%time

with open('../input/ships-in-satellite-imagery/shipsnet.json') as data_file:
    dataset = json.load(data_file)
shipsnet= pd.DataFrame(dataset)
print(shipsnet.shape)
print(shipsnet.columns)
print(shipsnet['data'][0:5])

data = np.array(dataset['data']).astype('uint8')
labels = np.array(dataset['labels']).astype('uint8')
data = data.reshape([-1, 3, 80, 80]).transpose([0,2,3,1])
print(shipsnet.shape)
%%time

data_prepared = pd.DataFrame(columns=['label', 'image'])

CULL_INTERVAL = 1  # For quick experiments, only use a subset of the images
FIGSIZE = 20
DISPLAY_IMAGES = True
DRAW_INTERVAL = 200  # How often should a sample image be shown

label_counts = Counter()
for img_idx, img_rgb in enumerate(data):
    if img_idx % CULL_INTERVAL != 0:
        continue
    if len(data_prepared) % DRAW_INTERVAL == 0:
        print("Progress: {:>8} of {:>8} {:>8}".format(img_idx, len(data), len(data_prepared)))
    
    label = labels[img_idx]
    label_counts[label] += 1
    
    # Convert to normalized grayscale
    img_gray = np.mean(img_rgb, axis=2)
    img_gray -= img_gray.min()
    img_gray /= img_gray.max() / 255
    
    # Calculate the magnitude spectrum in decibels, which seems to give the best orientation results
    img_fft = np.fft.fftshift(np.fft.fft2(img_gray))
    img_ms = 20 * np.log(np.abs(img_fft))
    
    # Mask the power spectrum to a circle as opposed to the default square so as to accumulate a fair amount of energy at all angles
    # Also, mask out the DC term. It's a distraction that can corrupt the orientation calculation.
    height_half_2 = (img_ms.shape[0] / 2)**2
    for y in range(80):
        for x in range(80):
            r2 = (y-40)**2 + (x-40)**2
            if (x == 40 and y == 40) or r2 > height_half_2:
                img_ms[y][x] = 0
    
    # Accumulate the value for every spectrum pixel into an angular bin.
    # The goal here is to find the "ray" through the center of the spectrum with the brightest accumulation of values.
    # That ray will indicate the image's primary orientation.
    height, width = img_ms.shape
    height_half, width_half = height / 2, width / 2
    height_quarter_2 = (height / 4)**2
    img_gray_polar = np.zeros([181, 181])
    img_gray_polar_counts = [0] * 181
    for y in range(height):
        for x in range(width):
            y_shift = round(y - height_half)
            x_shift = round(x - width_half)
            r2 = y_shift**2 + x_shift**2
            if r2 > height_quarter_2:
                continue
            angle = math.atan2(y_shift, x_shift) * (180. / math.pi) + 180
            if angle >= 180:
                angle -= 180  # For orientation, we only need 180 degrees, not 360
            angle_bin = round(angle)
            img_gray_polar[180][angle_bin] += img_ms[y][x]
            img_gray_polar_counts[angle_bin] += 1
    for x in range(181):
        if img_gray_polar_counts[x] > 0:
            img_gray_polar[180][x] /= img_gray_polar_counts[x]
    img_gray_polar /= img_gray_polar.max() / 255
    img_gray_polar = img_gray_polar.round()
    for x in range(181):
        for y in range(160, 180):
            img_gray_polar[y][x] = img_gray_polar[180][x]
        for yy in range(10):
            img_gray_polar[160 - yy - int(round(img_gray_polar[180][x] * (150 / 255)))][x] = img_gray_polar[180][x]
    max_loc = np.unravel_index(img_gray_polar.argmax(), img_gray_polar.shape)
    rot_angle_2 = max_loc[1]
    
    # This isn't very important, but there is no point in rotating more than 90 degrees
    if rot_angle_2 >= 90:
        rot_angle_2 = -180 + rot_angle_2
    
    # Rotate the image to align its central orientation vertically
    img_rot = scipy.ndimage.rotate(img_rgb, rot_angle_2, reshape=False, mode='nearest')
    
    # Recalculate the grayscale image
    img_rot_gray = np.mean(img_rot, axis=2)
    img_rot_gray -= img_rot_gray.min()
    img_rot_gray /= img_rot_gray.max() / 255
    
    # Downsample to 16x16 pixels
    img_rot_gray_16x16 = scipy.ndimage.zoom(img_rot_gray, .2, order=3)
    img_rot_gray_256 = img_rot_gray_16x16.flatten().round().astype(np.int64)
    img_rot_gray_256 -= img_rot_gray_256.min()
    img_rot_gray_256_norm = np.divide(img_rot_gray_256, np.max(img_rot_gray_256) / 255).astype(np.int64)
    
    # Store the prepared image
    data_prepared = data_prepared.append({'label': label, 'image': img_rot_gray_256_norm}, ignore_index=True)
        
    # Draw the images
    if DISPLAY_IMAGES and img_idx % DRAW_INTERVAL == 0:
        fig, (ax_rgb, ax_gray, ax_freq, ax_polar, ax_rot, ax_gray_16) = plt.subplots(1, 6)

        fig.set_figwidth(FIGSIZE)
        fig.set_figheight(FIGSIZE)
        
        ax_rgb.axis('off')
        ax_gray.axis('off')
        ax_freq.axis('off')
        ax_polar.axis('off')
        ax_rot.axis('off')
        ax_gray_16.axis('off')

        ax_rgb.imshow(img_rgb, vmax=255)
        ax_gray.imshow(img_gray, cmap='gray', vmax=255)
        ax_freq.imshow(img_ms, cmap='gray')
        ax_polar.imshow(img_gray_polar, cmap='rainbow', vmax=255)
        ax_rot.imshow(img_rot_gray, cmap='gray', vmax=255)
        ax_gray_16.imshow(img_rot_gray_16x16, cmap='gray', vmax=255)

print("Final data count: {}".format(len(data_prepared)))
print("Label counts: {}".format(label_counts.most_common()))
%%time

def shuffle(df, seed=None):
    return df.reindex(np.random.RandomState(seed=seed).permutation(df.index))

def split_train_test(df, random_state=None, test_size=0.2):
    test_data = pd.DataFrame(columns=['label', 'image'])
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train, test

def split_labels(df):
    df_true = df.loc[df['label'] == True]
    df_false = df.loc[df['label'] == False]
    return df_true, df_false

# Fix the random see so we can maintain consistency between repeated runs
RANDOM_SEED = 1

data_shuffled = shuffle(data_prepared, RANDOM_SEED)
data_train, data_test = split_train_test(data_shuffled, RANDOM_SEED)
data_train_true, data_train_false = split_labels(data_train)
data_test_true, data_test_false = split_labels(data_test)

print("PREPARED ({})".format(len(data_prepared)))
print("SHUFFLED ({})".format(len(data_shuffled)))
print("TRAIN ({})".format(len(data_train)))
print("TEST ({})".format(len(data_test)))
print("TRAIN TRUE ({})".format(len(data_train_true)))
print("TRAIN FALSE ({})".format(len(data_train_false)))
print("TEST TRUE ({})".format(len(data_test_true)))
print("TEST FALSE ({})".format(len(data_test_false)))
# Commented out since this demonstrative code cannot actually be run on Kaggle
# init_pyboard_serial_connection()
from timeit import default_timer

def train_network_method_1(start_start_id, max_rows, data):
    """
    Send training examples to the network, one by one.
    Positive examples will either be added to the network if they lie outside the active
    influence fields of all previously stored positive examples, or will be discarded.
    Negative examples will shrink the active influence fields of all overlapping stored
    positive examples.
    """
    start_time = default_timer()
    set_compression(True)
    
    verbose_level = get_verbose_level()
    print("< verbose_level: {}".format(verbose_level))
    
    # Send the training (learning) data
    columns = data.columns
    labels = data[columns[0]]
    rows = data[columns[1:]]
    
    labels_np = np.array(labels)
    rows_np = np.array(rows)
    
    print("< Num rows available: {}".format(len(rows_np)))
    
    for start_id in range(start_start_id, len(data), max_rows):
        status_found = False
        for attempt in range(0, 2):
            print("< " + "=" * 100)
            print("< start_id: {}".format(start_id))

            start_time_one_batch = default_timer()

            num_rows_sent = 0
            for i in range(len(rows_np)):
                if i < start_id:
                    continue
                if num_rows_sent >= max_rows:
                    break

                learn_one_pattern(i + 1, context, 1 if labels_np[i] else 0, rows_np[i][0], False)
                num_rows_sent += 1

            if verbose_level >= 1:
                print("< Num rows sent: {}".format(num_rows_sent))
                print("< " + "_" * 100)
                print("< Flushing last bundle")

            finalize_bundle()

            if verbose_level >= 2:
                print("< " + "_" * 100)
                print("< Resending messages as necessary")

            status_found, response = resend_messages_as_necessary("Learning pattern #{}".format(start_id + num_rows_sent), 1, "Learning pattern")
            print("< status_found: {}".format(status_found))
            if verbose_level >= 2:
                print("< Response:\n{}".format(response.replace("\n> ", "\n] ")))

            end_time_one_batch = default_timer()
            elapsed_time_one_batch = end_time_one_batch - start_time_one_batch
            print("< Elapsed time one batch: {}".format(elapsed_time_one_batch))
            
            if status_found:
                break
            else:
                print("< Stale, trying one more time.")
        if not status_found:
            break
    
    end_time = default_timer()
    elapsed_time = end_time - start_time
    print("< Elapsed time: {}".format(elapsed_time))
def write_true_data_to_network(start_start_id, end_id, max_rows, reset):
    """
    Write all positive training examples directly to the network, ignoring overlapping active
    influence fields.
    They will be stored with the default maximum AIF.
    The AIFs will later be shrunk in a second pass by exposing the network to negative
    training examples.
    
    Parameters:
        start_start_id (int): Training sample id from which to begin training enumeration
        end_id (int): Training sample id at which to end training enumeration
        max_rows (int): The PyBoard's serial buffer can overflow, so only send a few rows at a time
    """
    start_time = default_timer()
    
    set_compression(True)
    
    verbose_level = get_verbose_level()
    print("< verbose_level: {}".format(verbose_level))
    
    # Send the training (learning) data
    columns = data_train_true.columns
    labels = data_train_true[columns[0]]
    rows = data_train_true[columns[1:]]
    
    labels_np = np.array(labels)
    rows_np = np.array(rows)
    
    print("< Num rows available: {}".format(len(rows_np)))
    
    for start_id in range(start_start_id, end_id, max_rows):
        print("< " + "=" * 100)
        print("< start_id: {}".format(start_id))
        num_rows_sent = 0
        for i in range(len(rows_np)):
            if i < start_id:
                continue
            if num_rows_sent >= max_rows:
                break

            write_one_pattern(i + 1, context, 1 if labels_np[i] else 0, rows_np[i][0], reset)
            reset = False  # At most, only reset for the very first pattern
            num_rows_sent += 1

        if verbose_level >= 1:
            print("< Num rows sent: {}".format(num_rows_sent))
            print("< " + "_" * 100)
            print("< Flushing last bundle")

        finalize_bundle()

        if verbose_level >= 2:
            print("< " + "_" * 100)
            print("< Resending messages as necessary")

        status_found, response = resend_messages_as_necessary("Writing pattern #{}".format(start_id + num_rows_sent), 1, "Writing pattern")
        print("< status_found: {}".format(status_found))
        if verbose_level >= 2:
            print("< Response:\n{}".format(response.replace("\n> ", "\n] ")))
    
    end_time = default_timer()
    elapsed_time = end_time - start_time
    print("< Elapsed time: {}".format(elapsed_time))
%%time

start_id = 0
# Commented out since this demonstrative code cannot actually be run on Kaggle
# write_true_data_to_network(start_id, end_id=len(data_train_true), max_rows=50, reset=(start_id==0))
%%time

# Commented out since this demonstrative code cannot actually be run on Kaggle
# train_network_method_1(0, 100, data_train_false)
def get_classifications(start_id, max_num_rows_to_classify):
    """
    Send query patterns to the NM500 and get a response string back,
    but don't parse the classifications (labels) out of the string
    
    Parameters:
        start_id (int): Training sample id from which to begin query enumeration
        max_num_rows_to_classify (int): The PyBoard's serial buffer can overflow, so only send a few rows at a time
    """
    start_time = default_timer()
    
    rows_available = len(data_test)
    max_num_rows_to_classify = min(max_num_rows_to_classify, rows_available)
    verbose_level = get_verbose_level()
    if verbose_level >= 0:
        print("< Start id: {}".format(start_id))
        print("< Num rows to classify: {}".format(max_num_rows_to_classify))
    
    # Classify the test data
    queries = []
    columns = data_test.columns
    labels = data_test[columns[0]]
    rows = data_test[columns[1:]]
    labels_np = np.array(labels)
    rows_np = np.array(rows)
    
    for i in range(len(rows_np)):
        if i < start_id:
            continue

        label = labels_np[i]

        if verbose_level >= 2:
            print("< Classifying #{:>3}, Cxt {}, GT lbl {}".format(len(queries) + 1, context, label))
        
        classify_one_pattern(i + 1, context, rows_np[i][0])
        queries.append((context, label))

        if len(queries) >= max_num_rows_to_classify:
            break

    if verbose_level >= 1:
        print("< Flushing last bundle")
    
    finalize_bundle()

    status_found, response = resend_messages_as_necessary("Classification result #{}".format(start_id + len(queries)), 1, "Classifying pattern")
    print("< status_found: {}".format(status_found))
    if verbose_level >= 2:
        print("< Response:\n{}".format(response.replace("\n> ", "\n] ")))
    
    end_time = default_timer()
    elapsed_time = end_time - start_time
    print("< Classification elapsed time:   {:5.5}s ({:5.5}m)".format(elapsed_time, elapsed_time / 60.))
    
    return response, queries

def analyze_performance(response, queries, accumulating_result=None):
    """
    Parse query classifications (labels) out of a response string returned by the PyBoard as received from the NM500.
    Pair the labels up with the queries and, assuming the queries included ground-truth labels (i.e., test queries as opposed to in-the-wild queries),
    compute performance statistics of the network against the test dataset.
    
    Parameters:
        accumulating_result (list): Since we must send queries in small batches to protect the PyBoard's serial buffer from overflowing,
            we must accumulate the results from multiple batches together for final statistical analysis.
    """
    verbose_level = get_verbose_level()
    if verbose_level >= 3 or "Exception" in response or "Error" in response:
        print(response)
    
    groups = response.split("Classification result")
    
    num_queries = len(queries)

    num_test_events = accumulating_result[0] if accumulating_result else 0
    num_test_nonevents = accumulating_result[1] if accumulating_result else 0
    tp = accumulating_result[3] if accumulating_result else 0
    tn = accumulating_result[4] if accumulating_result else 0
    fp = accumulating_result[5] if accumulating_result else Counter()
    fn = accumulating_result[6] if accumulating_result else Counter()
    
    for i, group in enumerate(groups):
        if i == 0:
            pass
        if i > num_queries:
            print("Too many groups: {}".format(group))
            continue
        query = queries[i - 1]

        lines = group.split('\n') if response else []
        for line in lines:
            if "Category=" in line:
                if query[1] != 0:
                    num_test_events += 1
                else:
                    num_test_nonevents += 1

                words = line.split()
                category = int(words[6])
                if category == 65535:
                    category = 0
                elif category & 1 << 15:
                    # NM500's degenerate neuron flag (fully shrunk AIF)
                    category &= 0x7FFF
                reverse_label = category
                if reverse_label == query[1]:
                    if query[1] == 0:
                        tn += 1
                        if verbose_level >= 2:
                            print("< #{:>3}    Cxt {}    GT lbl {} == {} (cat {})   TN".format(i, query[0], query[1], reverse_label, category))
                    else:
                        tp += 1
                        if verbose_level >= 2:
                            print("< #{:>3}    Cxt {}    GT lbl {} == {} (cat {}) TP".format(i, query[0], query[1], reverse_label, category))
                else:
                    if query[1] == 0:
                        fp[(query[1], reverse_label)] += 1
                        if verbose_level >= 2:
                            print("< #{:>3}    Cxt {}    GT lbl {} != {} (cat {})     FP".format(i, query[0], query[1], reverse_label, category))
                    else:
                        fn[(query[1], reverse_label)] += 1
                        if verbose_level >= 2:
                            print("< #{:>3}    Cxt {}    GT lbl {} != {} (cat {})       FN".format(i, query[0], query[1], reverse_label, category))

    if accumulating_result:
        num_queries += accumulating_result[2]
                            
    fp_total = sum(fp.values())
    fn_total = sum(fn.values())
    accuracy = (tp + tn) / num_queries if num_queries != 0 else math.nan
    error = (fp_total + fn_total) / num_queries if num_queries != 0 else math.nan
    precision = tp / (tp + fp_total) if (tp != 0 or fp_total != 0) else math.nan
    recall = tp / (tp + fn_total) if (tp != 0 or fn_total != 0) else math.nan
    specificity = tn / (tn + fp_total) if (tn != 0 or fp_total != 0) else math.nan

    result = (num_test_events, num_test_nonevents, num_queries, tp, tn, fp, fn, accuracy, error, precision, recall, specificity)
    return result
    
def classify(start_id, max_num_rows_to_classify, accumulating_result=None):
    """
    Classify a set of queries, analyze the performance of those classifications against test ground-truth labels,
    and accumulate statistical performance metrics.
    """
    response, queries = get_classifications(start_id, max_num_rows_to_classify)
    result = analyze_performance(response, queries, accumulating_result)
    return result
%%time

start_start_id = 0
max_rows = 200  # The PyBoard's serial buffer can overflow, so only send a few rows at a time

accumulating_result = None

for start_id in range(start_start_id, len(data_test), max_rows):
    start_time = default_timer()

    print("< " + "=" * 100)
    print("< start_id: {}".format(start_id))
    
    # Commented out since this demonstrative code cannot actually be run on Kaggle
    # accumulating_result = classify(start_id, max_rows, accumulating_result)
    print("< Accumulating result: {}".format(accumulating_result))
    
    end_time = default_timer()
    elapsed_time = end_time - start_time
    print("< One experiment elapsed time:   {:5.5}s ({:5.5}m)".format(elapsed_time, elapsed_time / 60.))

print()
num_test_events, num_test_nonevents, num_queries, tp, tn, fp, fn, accuracy, error, precision, recall, specificity \
    = accumulating_result

fp_total = sum(fp.values())
fn_total = sum(fn.values())

s = ""
#     s += "\n< Max    train events & nonevents: {:>5}, {:>5}    (user-specified max, but may exceed data availability)".format(max_events, max_nonevents)
#     s += "\n< Actual train events & nonevents: {:>5}, {:>5}    ({:>5} total)".format(num_train_events, num_train_nonevents, num_train_events + num_train_nonevents)
#     s += "\n< Max and actual classifications:  {:>5}           ({:>5} actual)".format(num_classifications, num_queries)
s += "\n< Actual test events & nonevents: {} + {} = {} total".format(num_test_events, num_test_nonevents, num_test_events + num_test_nonevents)
s += "\n"
s += "\n< TP:          {:>5} (out of {})".format(tp, num_queries)
s += "\n< TN:          {:>5} (out of {})".format(tn, num_queries)
s += "\n< FP:          {:>5} (out of {})".format(fp_total, num_queries)
s += "\n< FN:          {:>5} (out of {})".format(fn_total, num_queries)
s += "\n"
for k, v in fp.items():
    s += "\n< FP ({}) => {}: {:>5} (out of {})".format('T' if k[0] else 'F', k[1], v, num_queries)
for k, v in fn.items():
    s += "\n< FN ({}) => {}: {:>5} (out of {})".format('T' if k[0] else 'F', k[1], v, num_queries)
s += "\n"
s += "\n< TP+TN:       {:>5} (out of {}, correct overall, aka accuracy)".format(tp + tn, num_queries)
s += "\n< FP+FN:       {:>5} (out of {}, incorrect overall, aka error)".format(fp_total + fn_total, num_queries)
s += "\n"
s += "\n< Accuracy:    {:>5.4}% [(TP+TN)/TOTAL] (correct classification regardless of pos./neg.)".format(accuracy * 100)
s += "\n< Error:       {:>5.4}% [(FP+FN)/TOTAL] (incorrect classification regardless of pos./neg.)".format(error * 100)
s += "\n< Precision:   {:>5.4}% [TP/(TP+FP)]    (pos. predictive value, or how informative is a TP prediction)".format(precision * 100)
s += "\n< Recall:      {:>5.4}% [TP/(TP+FN)]    (true pos. rate, % of pos. cases actually discovered, aka sensitivity & hit rate)".format(recall * 100)
s += "\n< Specificity: {:>5.4}% [TN/(TN+FP)]    (true neg. rate, % of neg. cases actually discovered)".format(specificity * 100)

print(s)
