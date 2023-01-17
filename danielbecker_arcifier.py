import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

from matplotlib import colors



import os

import json

import random

import warnings

import datetime
def get_filenames(directory):

    """ Get list with all files in the directory



        param directory: Path to the directory

        return result: List with filenames

    """   

    return [os.path.join(dirname, filename) for dirname, _, filenames in os.walk(directory) for filename in filenames]



def import_json_to_df(filename):

    """ Create a DataFrame from the JSON file

    

        Columns:

        - input: Input array of the Task (e.g. [[1]])

        - output: Output array of the Task (e.q. [[2]])

        - task_id: Filename (e.g. 'a1b2c3d4')

        - submission_id: Unique identifier for the test tasks (e.g. '{task_id}_0', '{task_id}_1')

        - type: train/test



        param filename: Path to the JSON file

        return result: DataFrame of the JSON file

    """ 

    with open(filename, 'r') as f:

        task = json.load(f)

    result = []

    for key in task:

        df = pd.DataFrame(task[key])

        df['task_id'] = os.path.splitext(os.path.basename(filename))[0]

        if key == 'test':

            df['submission_id'] = df.apply(lambda x: f"{x['task_id']}_{x.name}", axis=1)

        df['type'] = key

        result.append(df)

    result = pd.concat(result, ignore_index=True, sort=False)

    result['input'] = result['input'].apply(np.array)

    result['output'] = result['output'].apply(np.array)

    return result

    

data_dir = '/kaggle/input/abstraction-and-reasoning-challenge/'



submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))

training_files = sorted(get_filenames(os.path.join(data_dir, 'training')))

evaluation_files = sorted(get_filenames(os.path.join(data_dir, 'evaluation')))

test_files = sorted(get_filenames(os.path.join(data_dir, 'test')))





train = pd.concat([import_json_to_df(f) for f in training_files], ignore_index=True, sort=False)

validation = pd.concat([import_json_to_df(f) for f in evaluation_files], ignore_index=True, sort=False)

test = pd.concat([import_json_to_df(f) for f in test_files], ignore_index=True, sort=False)



train.shape, validation.shape, test.shape, submission.shape

print(f'Train: {train.shape}')

print(f'Validation: {validation.shape}')

print(f'Test: {test.shape}')   

print(f'Submission: {submission.shape}')
print('Example with 1 test pair')

display(train[train['task_id']=='007bbfb7'])

print('Example with 2 test pairs')

display(train[train['task_id']=='e9614598'])
warnings.filterwarnings('ignore')

pd.set_option('display.max_colwidth', -1)



# Random seeds

SEED = 13

random.seed(SEED)

np.random.seed(SEED)



# Max Size of images for padding

MAX_SIZE = 30

# Padding

PAD_ID = -1

PAD_COLOR = '#FFFFFF'

# Colors for Plots

ORG_COLOR_MAP = {0:'#000000', 1:'#0074D9', 2:'#FF4136', 3:'#2ECC40', 4:'#FFDC00',

         5:'#AAAAAA', 6:'#F012BE', 7:'#FF851B', 8:'#7FDBFF', 9:'#870C25'}

NUM_COLORS = len(ORG_COLOR_MAP)



COLOR_MAP = {PAD_ID:PAD_COLOR}

COLOR_MAP.update(ORG_COLOR_MAP)



DEFAULT_PREDICTION = [[0]]

DEFAULT_SUBMISSION = '|0| |0| |0|'



# Names of the DataFrame to use for plots or prints

DF_NAME = {id(train): 'train', id(validation): 'validation', id(test): 'test'}
def calculate_score(truth, pred1, pred2=None, pred3=None):

    """ Calculate Score for predictions

        

        Checks if truth value exists in one of the predictions

        

        param truth: List with truth output arrays

        param pred1: List with array for 1. predictions

        param pred2: List with array for 2. predictions

        param pred3: List with array for 3. predictions

        return score: Score of the model

    """

    result = []

    result.append([np.array_equal(truth[i], pred1[i]) for i in range(len(truth))])

    if pred2 is not None:

        result.append([np.array_equal(truth[i], pred2[i]) for i in range(len(truth))])

    if pred3 is not None:

        result.append([np.array_equal(truth[i], pred3[i]) for i in range(len(truth))])

    result = np.vstack(result)

    result = np.any(result, axis=0)

    score = (len(result)-np.sum(result))/len(result)

    print(f'Score: {np.round(score, 4)} (Found {np.sum(result)}/{len(result)})')

    return score



def prediction_to_submission(pred1, pred2=None, pred3=None):

    """ Creates the submission string from the three predictions.



        param pred1: Array with 1. prediction (e.g. [[1]])

        param pred2: Array with 2. prediction (default: None)

        param pred3: Array with 3. prediction (default: None)

        return result: String for submssion

    """   

    def list_to_str(pred):

        return '|'+'|'.join([''.join(map(str, r)) for r in pred])+'|'

    

    result = list_to_str(pred1)

    if pred2 is not None:

        result += ' '+list_to_str(pred2)

    else:

        result += ' '+list_to_str(DEFAULT_PREDICTION)

    if pred3 is not None:

        result += ' '+list_to_str(pred3)

    else:

        result += ' '+list_to_str(DEFAULT_PREDICTION)

    return result



def submission_to_prediction(sub):

    """ Creates the three predictions from the submission string



        param sub: Submission String (e.g. '|0| |1| |2|')

        return pred1: 1. prediction from submission string

        return pred2: 2. prediction from submission string

        return pred3: 3. prediction from submission string

    """  

    pred = [np.array([list(map(int, list(s))) for s in sub.split()[i].split('|')[1:-1]]) for i in range(3)]

    return pred[0], pred[1], pred[2]



def image_padding(img, size=(MAX_SIZE, MAX_SIZE), pad_int=-1):

    """ Creates a padding around the image



        param img: 2d array

        param size: max width and height (default (30, 30))

        param pad_int: integer used for the padding (default: -1)

        return result: padded image

    """ 

    top = int(np.ceil((size[0] - img.shape[0])/2))

    bottom = int(np.floor((size[0] - img.shape[0])/2))

    left = int(np.ceil((size[1] - img.shape[1])/2))

    right = int(np.floor((size[1] - img.shape[1])/2))

    return np.pad(img, ((top, bottom), (left, right)), constant_values=pad_int)



def remove_padding(img, pad_int=-1):

    """ Removes padding from image, if all values in row or column are the pad value



        param img: 2d array

        param pad_int: integer used for the padding (default: -1)

        return result: image

    """ 

    row_pad = np.all(img == pad_int, axis=1)

    img = img[~row_pad,:]

    col_pad = np.all(img == pad_int, axis=0)

    img = img[:,~col_pad]

    return img



def plot_task(df, task_id):

    """ Plot all inputs and outputs for the task



        param df: DataFrame with the input, output and task_id

        param task_id: id of the task to plot

    """ 

    cmap = colors.ListedColormap(COLOR_MAP.values())

    norm = colors.Normalize(vmin=-1, vmax=9)

    

    df_name = DF_NAME[id(df)]

    df = df[df['task_id'] == task_id].sort_values('type', ascending=False).reset_index()

    

    fig, axs = plt.subplots(2, df.shape[0], figsize=(3*df.shape[0],6))

    axs[0][0].text(0,-10, f'Task ID: {task_id} ({df_name})', fontsize=16)

    for idx, row in df.iterrows():

        input_img = row['input']

        input_img = image_padding(input_img)

        output_img = row['output']

        if np.isnan(output_img).any():

            output_img = np.array([[]])

        output_img = image_padding(output_img)

        task_type = row['type'].upper()

        axs[0][idx].imshow(input_img, cmap=cmap, norm=norm)

        #axs[0][idx].axis('off')

        axs[0][idx].set_yticklabels([])

        axs[0][idx].set_xticklabels([])

        axs[0][idx].set_title(f'{task_type}\nInput', color='green' if task_type == 'TRAIN' else 'blue', fontsize=16)

        axs[1][idx].imshow(output_img, cmap=cmap, norm=norm)

        #axs[1][idx].axis('off')

        axs[1][idx].set_yticklabels([])

        axs[1][idx].set_xticklabels([])

        axs[1][idx].set_title('Output', color='green' if task_type == 'TRAIN' else 'blue', fontsize=16)



    plt.tight_layout()

    plt.show()    
# Prediction to Submission Example

pred = np.array([[1,0,1], [0,1,0], [1,0,1], [1,1,1]])

print(f'Original Prediciton:\n{pred}')

pred_sub = prediction_to_submission(pred)

print(f'\nPrediction to Submission:\n{pred_sub}')

sub_pred = submission_to_prediction(pred_sub)[0]

print(f'\nSubmission to Prediction:\n{sub_pred}')
# Scoring Example

for df in [train, validation, test]:

    print(f'DataFrame: {DF_NAME[id(df)]}')

    score = calculate_score(df[df['type'] == 'test']['output'].values, [[[0]]]*df.shape[0], [[[1]]]*df.shape[0], [[[2]]]*df.shape[0])
# Plot Example

plot_task(train, train.loc[0, 'task_id'])

plot_task(validation, validation.loc[0, 'task_id'])

plot_task(test, test.loc[0, 'task_id'])
def get_features(row):

    def increased_height():

        """ Change between output and input height

        

            < 0: output is smaller

            = 0: output has same height

            > 0: output is higher

        """

        if row['type'] == 'test':

            return None

        else:

            return row['output'].shape[0] / row['input'].shape[0] - 1 

    def increased_width():

        """ Change between output and input width

        

            < 0: output is smaller

            = 0: output has same width

            > 0: output is larger

        """

        if row['type'] == 'test':

            return None

        else:

            return row['output'].shape[1] / row['input'].shape[1] - 1   

    def color_ratio(frame):

        """ Ratio of the colors



            param frame: 2d array

            return result: dictionary with ratio per color

        """

        if np.isnan(frame).any():

            return None

        else:

            result = {i:0.0 for i in range(NUM_COLORS)}

            unique, counts = np.unique(frame, return_counts=True)

            counts = counts / (frame.shape[0] * frame.shape[1])

            result.update(dict(zip(unique, counts)))

            return result

    def get_objects(img):

        """ Get objects from image

            

            -- under development --

            ToDo: Parameter if all or only one color per object

            ToDo: Check line img[start_y:end_y+1, start_x:end_x+1] = 0  # used to get only rectangles



            param img: 2d array

            return objects: dictionary with start and object array

        """

        if np.isnan(img).any():

            return None

        else:

            org = img.copy()

            img = image_padding(img, (img.shape[0]+2, img.shape[1]+2), 0)

            objects = []

            while True:

                start = np.where(img != 0)

                if len(start[0]) == 0:

                    break

                start_x = start[1][0]

                start_y = start[0][0]  

                end_x = start_x

                end_y = start_y

                control = set([(start_y, start_x)])

                while len(control) > 0:

                    c = control.pop()

                    act_y = c[0]

                    act_x = c[1]

                    if act_y > end_y:  end_y = act_y

                    if start_y > act_y:  start_y = act_y

                    if act_x > end_x:  end_x = act_x

                    if start_x > act_x:  start_x = act_x

                    img[act_y, act_x] = 0

                    sub = img[act_y-1:act_y+2, act_x-1: act_x+2]

                    for n in zip(np.where(sub)[0], np.where(sub)[1]):

                        n = (n[0]-1+act_y, n[1]-1+act_x)

                        img[n[0], n[1]] = 0

                        control = control | set([n])

                img[start_y:end_y+1, start_x:end_x+1] = 0

                start_y -= 1; end_y -= 1; start_x -= 1; end_x -= 1

                obj = org[start_y:end_y+1, start_x:end_x+1]

                objects.append({'start': (start_y, start_x), 'obj':obj})

            return objects

        

    def has_vertical_split(frame_in=row['input'], frame_out=row['output']):

        """ Checks if the input has a vertical split line

            and the output is part of this



            param frame_in: 2d array to check the split (input)

            param frame_out: 2d array to check if same size like split (output)

            return result: Boolean

        """

        if row['type'] == 'test':

            return False

        if (np.isnan(frame_in).any()) | (np.isnan(frame_out).any()):

            return False

        else:

            if frame_in.shape[1] % 2 == 0:

                return False

            mid = int(np.floor(frame_in.shape[1] / 2))

            mid_color = np.unique(frame_in[:,mid])

            if mid_color.shape[0] > 1:

                return False

            mid_color = mid_color[0]

            if mid_color in np.delete(frame_in, mid, 1):

                return False

            if frame_out.shape[1] != mid:

                return False

            return True

    def has_horizontal_split(frame_in=row['input'], frame_out=row['output']):

        """ Checks if the input has a horizontal split line

            and the output is part of this



            param frame_in: 2d array to check the split (input)

            param frame_out: 2d array to check if same size like split (output)

            return result: Boolean

        """

        if row['type'] == 'test':

            return False

        if (np.isnan(frame_in).any()) | (np.isnan(frame_out).any()):

            return False

        else:

            if frame_in.shape[0] % 2 == 0:

                return False

            mid = int(np.floor(frame_in.shape[0] / 2))

            mid_color = np.unique(frame_in[mid,:])

            if mid_color.shape[0] > 1:

                return False

            mid_color = mid_color[0]

            if mid_color in np.delete(frame_in, mid, 0):

                return False

            if frame_out.shape[0] != mid:

                return False

            return True

    def is_subset(frame, subframe):

        """ Checks if subframe is part of frame



            param frame: 2d array (full frame)

            param subframe: 2d array (sub frame)

            return result: Boolean

        """

        if row['type'] == 'test':

            return False

        if (np.isnan(frame).any()) | (np.isnan(subframe).any()):

            return False

        else:

            if (subframe.shape[0] > frame.shape[0]) | (subframe.shape[1] > frame.shape[1]):

                return False

            for i in range(frame.shape[0]-subframe.shape[0]+1):

                for j in range(frame.shape[1]-subframe.shape[1]+1):

                    match = np.array_equal(subframe, frame[i:i+subframe.shape[0], j:j+subframe.shape[1]])

                    if match:

                        return match

            return False

    def color_change():

        """ Checks if input and output use same cordinates

        

            return result: Boolean

        """

        if row['type'] == 'test':

            return False

        else:

            frame1 = np.where(row['input'] > 0, 1, 0)

            frame2 = np.where(row['output'] > 0, 1, 0)

            return np.array_equal(frame1, frame2)

    def adapted_size():

        # ToDo: Check if other method is better

        """ Checks if input is grown or shrunken

        

            return result: Boolean

        """

        if row['type'] == 'test':

            return False

        else:

            adapted_input = np.resize(row['input'], row['output'].shape)

            adapted_input = np.array_equal(adapted_input, row['output'])

            return adapted_input





    

    result = {

        'increased_height': increased_height(),

        'increased_width': increased_width(),

        #'adapted_size': adapted_size(),

        'color_ratio_input': color_ratio(row['input']),

        'color_ratio_output': color_ratio(row['output']),

        'input_objects': get_objects(row['input']),

        'output_objects': get_objects(row['output']),        

        'color_change': color_change(),

        'has_vertical_split': has_vertical_split(),

        'has_horizontal_split': has_horizontal_split(),

        'input_in_output': is_subset(row['output'], row['input']),

        'output_in_input': is_subset(row['input'], row['output']),

    }

    return result



def get_features2(row):

    def check_fill_out():

        """ Check if output is input with additional values

        

            return result: Boolean

        """

        if (row['increased_width'] == 0) & (row['increased_height'] == 0):

            output = row['output'].copy()

            blanks = np.where(row['input'] == 0, True, False)

            output[blanks == True] = 0

            return np.array_equal(row['input'], output)

        else:

            return False

    def fill_out_mask_color():

        """ Check if output is input with additional values

        

            return result: Boolean

        """

        if (row['increased_width'] == 0) & (row['increased_height'] == 0):

            diff = np.unique(row['input'][row['input'] != row['output']])

            if len(diff) == 1:

                return diff[0]

            else:

                return -1

        else:

            return -1

    def is_most_common_color():

        """ Check if the output color is the most common input color

        

            return result: Boolean

        """

        if (row['increased_width'] == 0) & (row['increased_height'] == 0):

            mc_color = np.argmax(list(row['color_ratio_input'].values()))

            return np.sum(np.where(row['output'] == mc_color, 1, 0)) == row['output'].shape[0]*row['output'].shape[1]

        else:

            return False

    def unique_output_color():

        """ Get the color of the output if it is unique

        

            return result: Integer for color

        """

        if row['color_ratio_output'] is None:

            return -1

        else:

            max_color = np.argmax(list(row['color_ratio_output'].values())[1:])+1

            if len(np.unique(list(row['color_ratio_output'].values())[1:])) > 2:

                return 0

            else:

                return max_color

    

    result = {

        'fill_out': check_fill_out(),

        'fill_out_mask_color': fill_out_mask_color(),

        'is_most_common_color': is_most_common_color(),

        'unique_output_color': unique_output_color()

    }

    return result



def get_features3(row):

    def left_diagonal_symmetric(frame):

        """ Check diagonal symmetric of array (top left to bottom right)

        

            return result: Boolean

        """

        if row['type'] == 'test':

            return False

        else:

            return np.array_equal(frame, frame.T)

    def right_diagonal_symmetric(frame):

        """ Check diagonal symmetric of array (top right to bottom left)

        

            return result: Boolean

        """

        if row['type'] == 'test':

            return False

        else:

            return np.array_equal(frame, np.flip(frame).T)

    def horizontal_symmetric(frame):

        """ Check diagonal horizontal of array

        

            return result: Boolean

        """

        if row['type'] == 'test':

            return False

        else:

            return np.array_equal(frame, np.flipud(frame))

    def vertical_symmetric(frame):

        """ Check diagonal vertical of array

        

            return result: Boolean

        """

        if row['type'] == 'test':

            return False

        else:

            return np.array_equal(frame, np.fliplr(frame))

    

    result = {

        'input_left_diagonal_symmetric': left_diagonal_symmetric(row['input']),

        'input_right_diagonal_symmetric': right_diagonal_symmetric(row['input']),

        'input_horizontal_symmetric': horizontal_symmetric(row['input']),

        'input_vertical_symmetric': vertical_symmetric(row['input']),

        'output_left_diagonal_symmetric': left_diagonal_symmetric(row['output']),

        'output_right_diagonal_symmetric': right_diagonal_symmetric(row['output']),

        'output_horizontal_symmetric': horizontal_symmetric(row['output']),

        'output_vertical_symmetric': vertical_symmetric(row['output']),

    }

    return result



def same_array_size(df, task_id, column='input'):

    """ Checks if all training tasks have same input size



        param df: DataFrame with the task id

        param task_id: task_id

        paramt column: Column with array (input/output)

        return result: boolean

    """

    result = df[(df['type'] != 'test') & (df['task_id'] == task_id)][column].apply(lambda x: np.array(x).shape).unique()

    return True if len(result) == 1 else False
# Create Features 

for df in [train, validation, test]:

    start_time = datetime.datetime.now()

    # Features 1

    features = pd.DataFrame(df.apply(lambda x: get_features(x), axis=1).to_list())

    df[features.columns] = features

    # Features 2

    features = pd.DataFrame(df.apply(lambda x: get_features2(x), axis=1).to_list())

    df[features.columns] = features

    # Features 3

    features = pd.DataFrame(df.apply(lambda x: get_features3(x), axis=1).to_list())

    df[features.columns] = features

    # Features 4

    df['same_input_size'] = df['task_id'].apply(lambda x: same_array_size(df, x, 'input'))

    df['same_output_size'] = df['task_id'].apply(lambda x: same_array_size(df, x, 'output'))

    df['count_input_objects'] = df['input_objects'].apply(lambda x: x if x is None else len(x))

    df['count_output_objects'] = df['output_objects'].apply(lambda x: x if x is None else len(x))        

    end_time = datetime.datetime.now()

    print(f'{DF_NAME[id(df)]}: {end_time-start_time}')
# Fill None values in test data based on training data

def get_train_average(df, task_id, column):

    """ Get Average of training data for task and column

    

    """

    return df[(df['task_id'] == task_id) & (df['type'] != 'test')][column].mean()



def check_boolean(df, task_id, column):

    """ Checks if all training data have True for task and column

    

    """

    return df[(df['task_id'] == task_id) & (df['type'] != 'test')][column].min()



column_types = train.columns.to_series().groupby(train.dtypes).groups

    

for df in [train, validation, test]:

    start_time = datetime.datetime.now()

    df_tests = df[df['type'] == 'test']

    for column in column_types[np.dtype('float64')]:

        df.loc[df_tests.index, column] = df_tests['task_id'].apply(lambda x: get_train_average(df, x, column))

    for column in column_types[np.dtype('int64')]:

        df.loc[df_tests.index, column] = df_tests['task_id'].apply(lambda x: int(get_train_average(df, x, column)) if np.ceil(get_train_average(df, x, column)) == np.floor(get_train_average(df, x, column)) else -1)

    for column in column_types[np.dtype('bool')]:

        df.loc[df_tests.index, column] = df_tests['task_id'].apply(lambda x: check_boolean(df, x, column))

    end_time = datetime.datetime.now()

    print(f'{DF_NAME[id(df)]}: {end_time-start_time}')
def create_heatmap(values, x_labels, y_labels, title=None, x_label=None, y_label=None, figsize=(8, 8)):

    #values = np.round(values/np.sum(values), 4)*100

    

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(values, cmap='viridis')



    ax.set_xticks(np.arange(len(x_labels)))

    ax.set_yticks(np.arange(len(y_labels)))

    ax.set_xticklabels(x_labels)

    ax.set_yticklabels(y_labels)



    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")



    for i in range(len(y_labels)):

        for j in range(len(x_labels)):

            text = ax.text(j, i, None if values[i, j] == 0 else values[i, j] , ha="center", va="center", color="w")

    ax.set_title(title)

    ax.set_xlabel(x_label)

    ax.set_ylabel(y_label)

    fig.tight_layout()

    plt.show()    
# Use only train dataset with training data for eda

df = train

df = df[df['type'] == 'train']
for column in ['input', 'output']:

    df['width'] = df[column].apply(lambda x: x.shape[1])

    df['height'] = df[column].apply(lambda x: x.shape[0])

    plt_data = df.groupby(['width', 'height'])['task_id'].count()

    x_labels = range(1, df['width'].max()+1)

    y_labels = range(1, df['height'].max()+1)

    plt_data = np.array([[plt_data[(y, x)] if (y, x) in plt_data.index else 0 for x in x_labels] for y in y_labels ])

    create_heatmap(plt_data, x_labels, y_labels, f'{column} Size', 'Width', 'Height')
# Input size

df['incr_width'] = df['increased_width'].apply(lambda x: 'bigger' if x > 0 else 'equal' if x == 0 else 'smaller')

df['incr_height'] = df['increased_height'].apply(lambda x: 'bigger' if x > 0 else 'equal' if x == 0 else 'smaller')

plt_data = df.groupby(['incr_width', 'incr_height'])['task_id'].count()

x_labels = ['bigger', 'equal', 'smaller']

y_labels = ['bigger', 'equal', 'smaller']

plt_data = np.array([[plt_data[(y, x)] if (y, x) in plt_data.index else 0 for x in x_labels] for y in y_labels ])

create_heatmap(plt_data, x_labels, y_labels, None, 'Increased Height', 'Increased Width', figsize=(5, 5))
plt_data = pd.DataFrame(index=column_types[np.dtype('bool')])

plt_data['True per Task'] = [len(df[df[column]]) for column in column_types[np.dtype('bool')]]

plt_data['True group by Tasks'] = [len(df.groupby('task_id')[column].min()[df.groupby('task_id')[column].min()]) for column in column_types[np.dtype('bool')]]

plt_data
for column in ['input', 'output']:

    plt_data = pd.DataFrame(df[f'color_ratio_{column}'].to_list())

    plt_data = plt_data.mean().rename(column)

    plt_data.plot(kind='pie', autopct='%1.1f%%', figsize=(5, 5))

    plt.show()
columns = list(column_types[np.dtype('float64')])+list(column_types[np.dtype('int64')])+list(column_types[np.dtype('bool')])

plt_data = df[columns].corr()

create_heatmap(np.round(plt_data.values, 2), columns, columns, figsize=(12, 12))
# Get Output for test dataset from validation dataset

# Use this only to evaluate the public test dataset

test_idx = test[test['type']=='test'].index

test.loc[test_idx, 'output'] = test.loc[test_idx, 'submission_id'].apply(lambda x: validation[validation['submission_id'] == x]['output'].values[0] if x in validation['submission_id'].values else None)
def get_prediction_column(df, idx):

    """ Get the column name of an unused prediction for the row



        param df: DataFrame

        param idx: Index of the row

        return next_prediction: Column name ('prediction_1', 'prediction_2', 'prediction_3')

    """

    next_prediction = 'prediction_3'

    for idx, val in df.loc[idx, ['prediction_1', 'prediction_2', 'prediction_3']].iteritems():

        if np.array_equal(val, DEFAULT_PREDICTION) or val is None:

            next_prediction = idx

            break

    return next_prediction



# Set all Predictions to default [[0]]

for df in [train, validation, test]:

    df['prediction_1'] = [DEFAULT_PREDICTION] * df.shape[0]

    df['prediction_2'] = [DEFAULT_PREDICTION] * df.shape[0]

    df['prediction_3'] = [DEFAULT_PREDICTION] * df.shape[0]
def check_startpoint(frame_in, frame_out, startpoint):

    """ Check if frame exists in corner of seconde frame



        param frame_in: Array Pattern

        param frame_out: Array to check the corners

        param startpoint: Corner

        return boolean: True if match exists

    """

    if startpoint == 'top_left':

        return np.array_equal(frame_in, frame_out[:frame_in.shape[0],:frame_in.shape[1]])

    elif startpoint == 'top_right':

        return np.array_equal(frame_in, frame_out[:frame_in.shape[0],-frame_in.shape[1]:])

    elif startpoint == 'bottom_left':

        return np.array_equal(frame_in, frame_out[-frame_in.shape[0]:,:frame_in.shape[1]])

    elif startpoint == 'bottom_right':

        return np.array_equal(frame_in, frame_out[-frame_in.shape[0]:,-frame_in.shape[1]:])

    else:

        return False

    

def rotations(frame, horizontal, vertical, start, hmethod='repeat', vmethod='repeat'):

    """ Use diffrent rotations and flips on array



        param frame: 2d Array

        param horizontal: number of horizontal rotations

        param vertical: number of vertical rotations

        param start: Startpoint (e.g. top_left)

        param hmethod: used method for horizontal

        param vmethod: used method for vertical

        return result: Transformed 2d Array

    """

    result = frame

    tmp = frame   

    for f in range(int(horizontal)):

        if hmethod == 'repeat':

            tmp = tmp

        elif hmethod == 'flip':

            tmp = np.flip(tmp)

        elif hmethod == 'fliplr':

            tmp = np.fliplr(tmp)

        elif hmethod == 'flipud':

            tmp = np.flipud(tmp)

        elif hmethod == 'rot90_3' and frame.shape[0] == frame.shape[1]:

            tmp = np.rot90(tmp, 3)

        elif hmethod == 'rot90_1' and frame.shape[0] == frame.shape[1]:

            tmp = np.rot90(tmp, 1)

        else:

            tmp = tmp

        if 'left' in start:

            result = np.append(result, tmp, axis=1)

        elif 'right' in start:

            result = np.append(tmp, result, axis=1)

    tmp = result

    for f in range(int(vertical)):

        if vmethod == 'repeat':

            tmp = tmp

        elif vmethod == 'flip':

            tmp = np.flip(tmp)

        elif vmethod == 'flipud':

            tmp = np.flipud(tmp)

        elif vmethod == 'fliplr':

            tmp = np.fliplr(tmp)

        elif vmethod == 'rot90_2':

            tmp = np.rot90(tmp, 2)

        elif vmethod == 'test' and frame.shape[0] == frame.shape[1]:

            tmp = np.rot90(frame, 2)

            tmp = np.append(tmp, np.rot90(frame, 3), axis=1)

        else:

            tmp = tmp

        if 'top' in start:

            result = np.append(result, tmp, axis=0)

        elif 'bottom' in start:

            result = np.append(tmp, result, axis=0)

    return result
if True:

    for df in [train, validation, test]:

        tasks = df[(df['type'] == 'test') & (df['input_in_output']) & (np.floor(df['increased_height']) == np.ceil(df['increased_height'])) & (np.floor(df['increased_width']) == np.ceil(df['increased_width']))]['task_id'].unique()

        df_test = df[(df['type'] == 'test') & (df['task_id'].isin(tasks))].copy()

        for task in tasks:

            df_temp = df[(df['type'] != 'test') & (df['task_id'] == task)].copy()

            df_temp['top_left'] = df_temp.apply(lambda x: check_startpoint(x['input'], x['output'], 'top_left'), axis=1)

            df_temp['top_right'] = df_temp.apply(lambda x: check_startpoint(x['input'], x['output'], 'top_right'), axis=1)

            df_temp['bottom_left'] = df_temp.apply(lambda x: check_startpoint(x['input'], x['output'], 'bottom_left'), axis=1)

            df_temp['bottom_right'] = df_temp.apply(lambda x: check_startpoint(x['input'], x['output'], 'bottom_right'), axis=1)

            starts = df_temp[['top_left', 'top_right', 'bottom_left', 'bottom_right']].min()

            starts = list(starts[starts].index)

            if len(starts) == 0:

                df_test = df_test[df_test['task_id'] != task]

                continue

            found = None

            for start in starts:

                for hmethod in ['repeat', 'flip', 'fliplr', 'flipud', 'rot90_3', 'rot90_1']:

                        for vmethod in ['repeat', 'flip', 'fliplr', 'flipud', 'rot90_2', 'test']:

                            df_temp[f'pred_{start}'] = df_temp.apply(lambda x: rotations(x['input'], x['increased_width'], x['increased_height'], start, hmethod, vmethod), axis=1)

                            df_temp[f'pred_{start}'] = df_temp.apply(lambda x: np.array_equal(x[f'pred_{start}'], x['output']), axis=1)

                            if df_temp[f'pred_{start}'].min():

                                found = True

                                break

                        if found:

                            break

                if found:

                    for idx, row in df_test[df_test['task_id'] == task].iterrows():

                        prediction = rotations(row['input'], row['increased_width'], row['increased_height'], start, hmethod, vmethod)

                        df_test.loc[idx, get_prediction_column(df_test, idx)] = [prediction]

                    break

            #if not found:

            #    plot_task(df, task)



        print(f'\nDataFrame: {DF_NAME[id(df)]}')

        score = calculate_score(df_test['output'].values, df_test['prediction_1'].values, df_test['prediction_2'].values, df_test['prediction_3'].values)

        df.loc[df_test.index, ['prediction_1', 'prediction_2', 'prediction_3']] = df_test[['prediction_1', 'prediction_2', 'prediction_3']]
def split_frame(frame, split='vertical'):

    """ Splits one frame to two subframes

        

        param frame: 2d array

        param split: vertical/horizontal

        return frame1: 1. frame of split (left or upper)

        return frame2: 2. frame of split (right or below)

    """

    if split=='vertical':

        mid = int(np.floor(frame.shape[1]/2))

        frame1 = frame[:,:mid]

        frame2 = frame[:,mid+1:]

    else:

        mid = int(np.floor(frame.shape[0]/2))

        frame1 = frame[:mid,:]

        frame2 = frame[mid+1:,:]

    return frame1, frame2



def splitting(frame, split='vertical', aggmethod='min', flip_frame1=None, flip_frame2=None, color=None):

    """ Use diffrent rotations and aggregation methods on splitted arrays



        param frame: 2d Array with split

        param split: kind of split (horizontal, vertical)

        param aggmethod: Method to merge the two splitted arrays

        param flip_frame1: Method for rotations on first frame

        param flip_frame2: Method for rotations on second frame

        param color: Use unique color for output

        return result: Transformed 2d Array

    """

    frame1, frame2 = split_frame(frame, split)

    if flip_frame1 == 'flip':

        frame1 = np.flip(frame1)

    elif flip_frame1 == 'fliplr':

        frame1 = np.fliplr(frame1)

    elif flip_frame1 == 'flipud':

        frame1 = np.flipud(frame1)

        

    if flip_frame2 == 'flip':

        frame2 = np.flip(frame2)

    elif flip_frame2 == 'fliplr':

        frame2 = np.fliplr(frame2)

    elif flip_frame2 == 'flipud':

        frame2 = np.flipud(frame2)

    

    if aggmethod == 'min':

        result = np.min(np.dstack((frame1, frame2)), axis=2)

    elif aggmethod == 'max':

        result = np.max(np.dstack((frame1, frame2)), axis=2)

    elif aggmethod == 'none_match':

        result = frame1 + frame2

        result = np.where(result > np.max(np.append(frame1, frame2)), 0, result)

    elif aggmethod == 'merge_2_in_1':

        result = frame1 + frame2

        if np.max(result) > np.max(np.append(frame1, frame2)):

            result = frame1

    elif aggmethod == 'merge_1_in_2':

        result = frame1 + frame2

        if np.max(result) > np.max(np.append(frame1, frame2)):

            result = frame2

    elif aggmethod == 'blanks':

        result = np.max(np.dstack((frame1, frame2)), axis=2)

        result = np.where(result == 0, 1, 0)

    if color:

        result[result > 0] = color

    return result
if True:

    for df in [train, validation, test]:

        tasks = df[(df['type'] == 'test') & ((df['has_vertical_split']) | (df['has_horizontal_split']))]['task_id'].unique()

        df_test = df[(df['type'] == 'test') & (df['task_id'].isin(tasks))].copy()

        for task in tasks:

            df_temp = df[(df['type'] != 'test') & (df['task_id'] == task)].copy()

            if df_temp['unique_output_color'].min() == df_temp['unique_output_color'].max():

                output_color = df_temp['unique_output_color'].values[0]

                if output_color <= 0:

                    output_color = None

            else:

                output_color = None

            found = None

            for aggmethod in ['min', 'max', 'blanks', 'none_match', 'merge_2_in_1', 'merge_1_in_2']:

                for flip_frame1 in [None, 'flip', 'fliplr', 'flipud']:

                    for flip_frame2 in [None, 'flip', 'fliplr', 'flipud']:



                        if output_color:

                            df_temp['pred'] = df_temp.apply(lambda x: splitting(x['input'], 'vertical' if x['has_vertical_split'] else 'horizontal', aggmethod, flip_frame1, flip_frame2, output_color), axis=1)

                        else:

                            df_temp['pred'] = df_temp.apply(lambda x: splitting(x['input'], 'vertical' if x['has_vertical_split'] else 'horizontal', aggmethod, flip_frame1, flip_frame2), axis=1)



                        df_temp['pred'] = df_temp.apply(lambda x: np.array_equal(x['pred'], x['output']), axis=1)

                        if df_temp[f'pred'].min():

                            found = True

                            break

                    if found:

                        break

                if found:

                    break



            if found:

                for idx, row in df_test[df_test['task_id'] == task].iterrows():

                    if output_color:

                        prediction = splitting(row['input'], 'vertical' if row['has_vertical_split'] else 'horizontal', aggmethod, flip_frame1, flip_frame2, output_color)

                    else:

                        prediction = splitting(row['input'], 'vertical' if row['has_vertical_split'] else 'horizontal', aggmethod, flip_frame1, flip_frame2)

                    df_test.loc[idx, get_prediction_column(df_test, idx)] = [prediction]

            #if not found:

            #    plot_task(df, task)



        print(f'\nDataFrame: {DF_NAME[id(df)]}')

        score = calculate_score(df_test['output'].values, df_test['prediction_1'].values, df_test['prediction_2'].values, df_test['prediction_3'].values)

        df.loc[df_test.index, ['prediction_1', 'prediction_2', 'prediction_3']] = df_test[['prediction_1', 'prediction_2', 'prediction_3']]
def fill_out(frame, method, mask_color=0):

    """ fill out missing values with a mirrored version of itself 

        

        param frame: 2d array

        param method: used symmetry

        return frame: filled image

    """

    mask = np.where(frame == mask_color)

    if method == 'output_left_diagonal_symmetric':

        transpose = frame.T

    elif method == 'output_right_diagonal_symmetric':

        transpose = np.flip(frame).T

    elif method == 'output_horizontal_symmetric':

        transpose = np.flipud(frame)

    elif method == 'output_vertical_symmetric':

        transpose = np.fliplr(frame)

    for i in range(len(mask[0])):

        frame[mask[0][i], mask[1][i]] = transpose[mask[0][i], mask[1][i]]

    return frame
if True:

    for df in [train, validation, test]:

        tasks = df[df['type'] == 'test'][df['fill_out_mask_color'] >= 0][(df['output_left_diagonal_symmetric']) | (df['output_right_diagonal_symmetric']) | (df['output_horizontal_symmetric']) | (df['output_vertical_symmetric'])]['task_id'].unique()

        df_test = df[(df['type'] == 'test') & (df['task_id'].isin(tasks))].copy()

        for task in tasks:

            found = None

            df_temp = df[(df['type'] != 'test') & (df['task_id'] == task)].copy()

            df_temp['pred_output'] = df_temp['input']

            methods = df_temp[['output_left_diagonal_symmetric', 'output_right_diagonal_symmetric', 'output_horizontal_symmetric', 'output_vertical_symmetric']].min()

            methods = list(methods[methods].index)

            for method in methods*2:

                df_temp['pred_output'] = df_temp.apply(lambda x: fill_out(x['pred_output'], method, x['fill_out_mask_color']), axis=1)

                df_temp['pred'] = df_temp.apply(lambda x: np.array_equal(x['pred_output'], x['output']), axis=1)

                if df_temp[f'pred'].min():

                    found = True

                    break

                    

            # makes prediction independent from training data... 

            # correct way: if found: 

            if True:

                for idx, row in df_test[df_test['task_id'] == task].iterrows():

                    prediction = row['input'].copy()

                    for method in methods*2:

                        prediction = fill_out(prediction, method, row['fill_out_mask_color'])

                    df_test.loc[idx, get_prediction_column(df_test, idx)] = [prediction]        

            #if not found:

            #    plot_task(df, task)



        print(f'\nDataFrame: {DF_NAME[id(df)]}')

        score = calculate_score(df_test['output'].values, df_test['prediction_1'].values, df_test['prediction_2'].values, df_test['prediction_3'].values)

        df.loc[df_test.index, ['prediction_1', 'prediction_2', 'prediction_3']] = df_test[['prediction_1', 'prediction_2', 'prediction_3']]
def grow_array(frame, growing_shape):

    """ Grow array according to the shape (height and width)

        

        param frame: 2d array

        param growing_shape: shape (height and width)

        return frame: grown array

    """

    frame = np.vstack([np.hstack([np.tile(frame[r, c], growing_shape) for c in range(frame.shape[1])]) for r in range(frame.shape[0])])

    return frame



if True:

    for df in [train, validation, test]:

        tasks = df[(df['type'] == 'test')

               & (df['increased_height'].apply(np.floor) == df['increased_height'].apply(np.ceil)) 

               & (df['increased_width'].apply(np.floor) == df['increased_width'].apply(np.ceil)) 

               & (df['increased_height'] == df['increased_width']) 

               & (df['increased_height'] > 0) & (df['increased_width'] > 0)]['task_id'].unique()

        df_test = df[(df['type'] == 'test') & (df['task_id'].isin(tasks))].copy()

        for task in tasks:

            df_temp = df[(df['type'] != 'test') & (df['task_id'] == task)].copy()

            df_temp['pred_output'] = df_temp['input']



            df_temp['pred'] = df_temp.apply(lambda x: grow_array(x['input'], (int(x['increased_height'])+1, int(x['increased_width'])+1)), axis=1)

            df_temp['pred'] = df_temp.apply(lambda x: np.array_equal(x['pred'], x['output']), axis=1)

            if df_temp[f'pred'].min():

                for idx, row in df_test[df_test['task_id'] == task].iterrows():

                    prediction = grow_array(row['input'], (int(row['increased_height'])+1, int(row['increased_width'])+1))

                    df_test.loc[idx, get_prediction_column(df_test, idx)] = [prediction]

            #else:

            #    plot_task(df, task)

                

        print(f'\nDataFrame: {DF_NAME[id(df)]}')

        score = calculate_score(df_test['output'].values, df_test['prediction_1'].values, df_test['prediction_2'].values, df_test['prediction_3'].values)

        df.loc[df_test.index, ['prediction_1', 'prediction_2', 'prediction_3']] = df_test[['prediction_1', 'prediction_2', 'prediction_3']]                
def get_color_mapping(frame_in, frame_out, color_mapping=None):

    """ Get Dictionary with color change

        

        param frame_in: 2d array Input

        param frame_out: 2d array Output with new colors

        param color_mapping: existing start mapping

        return color_mapping: Dictionary with color change {old_color: new_color}

    """

    if color_mapping == False:

        return False

    if color_mapping is None:

        color_mapping = dict(zip(range(NUM_COLORS), range(NUM_COLORS)))

    for key in color_mapping.keys():

        val = color_mapping[key]

        new_color = frame_out[np.where(frame_in == key)]

        new_color = np.unique(new_color)   

        if len(new_color) > 1:

            return False

        elif len(new_color) == 1:

            new_color = new_color[0]

            if key == val:

                color_mapping[key] = new_color

            elif val != new_color:

                return False

    return color_mapping



def change_colors(frame, color_mapping):

    """ Changes the colors according to the mapping

        

        param frame: 2d array Input

        param color_mapping: Dictionary with color mapping {old_color: new_color}

        return result: recolored frame

    """

    result = frame.copy()

    for key in color_mapping.keys():

        val = color_mapping[key]

        result[np.where(frame == key)] = val

    return result
if True:

    for df in [train, validation, test]:

        tasks = df[(df['type'] == 'test') & (df['color_change'])]['task_id'].unique()

        df_test = df[(df['type'] == 'test') & (df['task_id'].isin(tasks))].copy()

        for task in tasks:

            color_mapping = dict(zip(range(NUM_COLORS), range(NUM_COLORS)))

            df_temp = df[(df['type'] != 'test') & (df['task_id'] == task)].copy()

            df_temp['pred'] = df_temp.apply(lambda x: get_color_mapping(x['input'], x['output']), axis=1)

            for pred_color_mapping in df_temp['pred'].to_list():

                if pred_color_mapping == False:

                    color_mapping = False

                    break

                for key in color_mapping.keys():

                    val = color_mapping[key]

                    new_color = pred_color_mapping[key]

                    if key == val and key != new_color:

                        color_mapping[key] = new_color

                    elif val != new_color and key != new_color:

                        color_mapping = False

                        break

                if color_mapping == False:

                    break

            if color_mapping == False:

                continue 

            df_temp['pred'] = df_temp['input'].apply(lambda x: change_colors(x, color_mapping))

            df_temp['pred'] = df_temp.apply(lambda x: np.array_equal(x['pred'], x['output']), axis=1)

            if df_temp[f'pred'].min():

                for idx, row in df_test[df_test['task_id'] == task].iterrows():

                    prediction = change_colors(row['input'], color_mapping)

                    df_test.loc[idx, get_prediction_column(df_test, idx)] = [prediction]

                    

        print(f'\nDataFrame: {DF_NAME[id(df)]}')

        score = calculate_score(df_test['output'].values, df_test['prediction_1'].values, df_test['prediction_2'].values, df_test['prediction_3'].values)

        df.loc[df_test.index, ['prediction_1', 'prediction_2', 'prediction_3']] = df_test[['prediction_1', 'prediction_2', 'prediction_3']] 
def object_detection(objects, method):

    """ Check extracted input objects

        

        param objects: extracted objects from input

        param method: used method to filter objects

        return result: object

    """

    objects = [v['obj'] for v in objects]

    if method == 'most_common_obj':

        str_objects = [str(obj.reshape(-1,)) for obj in objects]

        uobj, ucount = np.unique(str_objects, return_counts=True)

        str_obj = uobj[np.argmax(ucount)]

        result = objects[str_objects.index(str_obj)]

    elif method == 'rarest_obj':

        str_objects = [str(obj.reshape(-1,)) for obj in objects]

        uobj, ucount = np.unique(str_objects, return_counts=True)

        str_obj = uobj[np.argmin(ucount)]

        result = objects[str_objects.index(str_obj)]

    elif method == 'rarest_color':

        colors = [x for o in objects for x in np.unique(o) if x != 0]

        ucol, ucount = np.unique(colors, return_counts=True)

        if len(colors) == len(objects) and len(np.where(ucount == 1)) == 1 and len(np.where(ucount == 1)[0]) != 0:

            result = objects[colors.index(ucol[np.where(ucount == 1)][0])]

        else:

            result = objects[0]

    elif method == 'smallest':

        sizes = [x.shape[0]*x.shape[1] for x in objects]

        result = objects[np.argmin(sizes)]

    elif method == 'biggest':

        sizes = [x.shape[0]*x.shape[1] for x in objects]

        result = objects[np.argmax(sizes)]

    return result
if True:

    for df in [train, validation, test]:

        tasks = df[(df['type'] == 'test') & (df['output_in_input'])]['task_id'].unique()

        df_test = df[(df['type'] == 'test') & (df['task_id'].isin(tasks))].copy()

        for task in tasks:

            df_temp = df[(df['type'] != 'test') & (df['task_id'] == task)].copy()

            found = None

            for method in ['most_common_obj', 'rarest_obj', 'rarest_color', 'smallest', 'biggest']:

                df_temp['pred'] = df_temp.apply(lambda x: object_detection(x['input_objects'], method), axis=1)



                df_temp['pred'] = df_temp.apply(lambda x: np.array_equal(x['pred'], x['output']), axis=1)

                if df_temp[f'pred'].min():

                    found = True

                    break

            if found:

                for idx, row in df_test[df_test['task_id'] == task].iterrows():

                    prediction = object_detection(row['input_objects'], method)

                    df_test.loc[idx, get_prediction_column(df_test, idx)] = [prediction]  

            #if not found:

            #    plot_task(df, task)     

  

        print(f'\nDataFrame: {DF_NAME[id(df)]}')

        score = calculate_score(df_test['output'].values, df_test['prediction_1'].values, df_test['prediction_2'].values, df_test['prediction_3'].values)

        df.loc[df_test.index, ['prediction_1', 'prediction_2', 'prediction_3']] = df_test[['prediction_1', 'prediction_2', 'prediction_3']] 
for df in [train, validation, test]:

    print(f'DataFrame: {DF_NAME[id(df)]}')

    df['match'] = df.apply(lambda x: np.array_equal(x['output'], x['prediction_1']) | np.array_equal(x['output'], x['prediction_2']) | np.array_equal(x['output'], x['prediction_3']), axis=1)

    df = df[df['type'] == 'test']

    score = calculate_score(df['output'].values, df['prediction_1'].values, df['prediction_2'].values, df['prediction_3'].values)
df = test[test['type'] == 'test'].copy()

df['submission'] = df.apply(lambda x: prediction_to_submission(x['prediction_1'], x['prediction_2'], x['prediction_3']), axis=1)

df = df.set_index('submission_id')['submission']

id_check = len(set(submission['output_id']) & set(df.index))

print(f'{df.shape[0]} Test ids \n{submission.shape[0]} Submssion ids \n{id_check} Intersection')



submission['output'] = submission['output_id'].apply(lambda x: df[x] if x in df.index else DEFAULT_SUBMISSION)

submission.to_csv('submission.csv', index=False)

submission[submission['output'] != DEFAULT_SUBMISSION]