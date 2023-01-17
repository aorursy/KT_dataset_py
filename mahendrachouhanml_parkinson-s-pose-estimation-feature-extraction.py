try:

    import ujson as json

except ImportError:

    try:

        import simplejson as json

    except ImportError:

        import json

import pickle

import numpy as np

from ipykernel import kernelapp as app

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import signal

from scipy import stats

from scipy.integrate import simps

from scipy.spatial import ConvexHull

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier



sns.set(font_scale=1.2)
trajectory_file = '../input/parkinsons-visionbased-pose-estimation-dataset/UDysRS_UPDRS_Export/Communication_all_export.txt'



with open(trajectory_file, 'r') as infile:

    comm_dict = json.load(infile)



print("total number of records :{}".format(len(comm_dict.keys())))
print("Key in Dictionary:",comm_dict['26-1'].keys())

sorted(comm_dict['26-1']['position'].keys())
%matplotlib inline

part = "Lank"

horizontal_displacemt_array = np.array(comm_dict['26-1']['position'][part])[:,0]

vertical_displacement_array = np.array(comm_dict['26-1']['position'][part])[:,1]

fig = plt.figure(figsize=(15, 4))

ax = fig.add_subplot(121)

ax.plot(horizontal_displacemt_array,label='Horizental motion')

ax.legend(loc='best')



ax2 = fig.add_subplot(122)

ax2.plot(vertical_displacement_array,label='vertical motion')

ax2.legend(loc='best')



plt.suptitle('Joint motion')

fig.show()
def combinied_horizental_and_vertical(horizental_array, vertical_array):

    combined_array = np.sqrt(np.square(horizental_array) + np.square(vertical_array))

    return combined_array
def convert_into_velocity(dispacement_array, plot=False):

    velocity_array = np.diff(dispacement_array)

    return velocity_array
def convert_into_acceleration(velocity_array, plot=False):

    accelation_array = np.diff(velocity_array)

    return accelation_array
def convert_into_jerk(accelation_array, plot=False):

    jerk_array = np.diff(accelation_array)

    return jerk_array
def get_kinetic_feature(motion):

    max_motion = np.amax(motion, axis=0)

    median_motion = np.median(motion)

    mean_motion = np.mean(motion, axis=0)

    standard_division_motion = np.std(motion)

    IQR_range = stats.iqr(motion, interpolation = 'midpoint')

    return [max_motion, median_motion, mean_motion, standard_division_motion, IQR_range]

#print(get_kinetic_feature(displacemt_array))
def get_spectral_feature(signals, sample_frequancy=10, is_plot=False):

    sf = sample_frequancy

    win = 4 * sf

    

    # calcutate the Spectral entropy.

    def spectral_entropy(psd, normalize=False):

        psd_norm = np.divide(psd, psd.sum())

        se = -np.multiply(psd_norm, np.log2(psd_norm)).sum()

        if normalize:

            se /= np.log2(psd_norm.size)

        return se

    

    # calculate the power band for given frequancy.

    def bandpower(psd, freqs, min_freqs, max_freqs, is_plot=False):

        # Define delta lower and upper limits

        low, high = min_freqs, max_freqs



        # Find intersecting values in frequency vector

        idx_delta = np.logical_and(freqs >= low, freqs <= high)



        if is_plot:

            # Plot the power spectral density and fill the delta area

            plt.figure(figsize=(7, 4))

            plt.plot(freqs, psd, lw=2, color='k')

            plt.fill_between(freqs, psd, where=idx_delta, color='skyblue')

            plt.xlabel('Frequency (Hz)')

            plt.ylabel('Power spectral density (uV^2 / Hz)')

            plt.xlim([0, 10])

            plt.ylim([0, psd.max() * 1.1])

            plt.title("Welch's periodogram")

            sns.despine()



        # Frequency resolution

        freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25



        # Compute the absolute power by approximating the area under the curve

        delta_power = simps(psd[idx_delta], dx=freq_res)

        #print('Absolute delta power: %.3f uV^2' % delta_power)

        return delta_power



    freqs, psd = signal.welch(signals, sf, nperseg=win)

    if is_plot:

        sns.set(font_scale=1.2, style='white')

        plt.figure(figsize=(8, 4))

        plt.plot(freqs, psd, color='k', lw=2)

        plt.xlabel('Frequency (Hz)')

        plt.ylabel('Power spectral density (V^2 / Hz)')

        plt.ylim([0, psd.max() * 1.1])

        plt.title("Welch's periodogram")

        plt.xlim([0, freqs.max()])

        sns.despine()

    #print(dir(psd))

    features = {}

    features["peak_magnitude"] = np.sqrt(psd.max())

    features["entropy"] = spectral_entropy(psd)

    features["half_point"] = freqs.mean()

    

    features["total_power"] = bandpower(psd, freqs, freqs.min(), freqs.max(), is_plot)

    features["power_bands_0.5_to_1"] = bandpower(psd, freqs, 0.5, 1, is_plot)

    features["power_bands_0_to_2"] = bandpower(psd, freqs, 0, 2, is_plot)

    features["power_bands_0_to_4"] = bandpower(psd, freqs, 0, 4, is_plot)

    features["power_bands_0_to_6"] = bandpower(psd, freqs, 0, 6, is_plot)

    return features

get_spectral_feature(vertical_displacement_array, is_plot=True)
# calculate the Area of Convex Hull of joint movement

def get_convexhull(darray):

    hull = ConvexHull(darray)

    return hull.area

get_convexhull(np.array(comm_dict['26-1']['position']["Lank"]))
def record_convertion(position_array, position_name, record_id="1-1"):

    position_array = np.array(position_array)

    horizantal_position = position_array[:, 0]

    vertical_position = position_array[:, 1]

    displacement_array = combinied_horizental_and_vertical(horizantal_position, vertical_position) 

    velocity_array = convert_into_velocity(displacement_array)

    accelation_array = convert_into_acceleration(velocity_array)

    jerk_array = convert_into_jerk(accelation_array)

    record = record_id.split("-")



    row = [record_id, int(record[0]), record[1], position_name]

    row.extend(get_kinetic_feature(velocity_array))

    row.extend(get_kinetic_feature(accelation_array))

    row.extend(get_kinetic_feature(jerk_array))

    spectral_feature_displacemt = get_spectral_feature(displacement_array)

    row.extend([value for key, value in spectral_feature_displacemt.items()])

    spectral_feature_velocity = get_spectral_feature(velocity_array)

    row.extend([value for key, value in spectral_feature_velocity.items()])

    convex_hull = get_convexhull(position_array)

    row.extend([convex_hull])

    return row
record_df = pd.DataFrame(columns=["combine_record_id","record_id", "term","position_name",

                                  "speed_max", "speed_median", "speed_mean", "speed_std_div", "speed_iqr_range",

                                 "acceleration_max", "acceleration_median", "acceleration_mean", "acceleration_std_div", "accelerati_iqr_range",

                                 "jerk_max", "jerk_median", "jerk_mean", "jerk_std_div", "jerk_iqr_range",

                                 "displacement_peak_magnitude","displacement_entropy", "displacement_half_point", "displacement_total_power",

                                 "displacement_power_bands_0.5_to_1","displacement_power_bands_0_to_2", "displacement_power_bands_0_to_4", "displacement_power_bands_0_to_6",

                                 "velocity_peak_magnitude","velocity_entropy", "velocity_half_point", "velocity_total_power",

                                 "velocity_power_bands_0.5_to_1","velocity_power_bands_0_to_2", "velocity_power_bands_0_to_4", "velocity_power_bands_0_to_6",

                                 "convexhull"])

index = 0



for record_id, values in comm_dict.items():

    positions = values["position"]

    resp = values["resp"]

    for position_name, position_array in positions.items():

        row = record_convertion(position_array, position_name, record_id)

        record_df.loc[index] = row

        index += 1



record_df.head(10)
rating_file = '../input/parkinsons-visionbased-pose-estimation-dataset/UDysRS_UPDRS_Export/UDysRS.txt'



with open(rating_file, 'r') as infile:

    ratings = json.load(infile)



ratings.keys()
ratings['Communication']['2']
sub_score_dict = {"Neck":["face"],

        "Larm":["Lsho", "Lelb", "Lwri"],

        "Rarm":["Rsho", "Relb", "Rwri"],

        "Trunk":["Rsho", "Lsho"],

        "Rleg":["Rhip", "Rkne", "Rank"],

        "Lleg":["Lhip", "Lkne", "Lank"]}



#sub_score_dict

groups = record_df.groupby("combine_record_id")

processed_df = pd.DataFrame(columns=["combine_record_id","record_id", "term","position_name","sub_score",

                                  "speed_max", "speed_median", "speed_mean", "speed_std_div", "speed_iqr_range",

                                 "acceleration_max", "acceleration_median", "acceleration_mean", "acceleration_std_div", "accelerati_iqr_range",

                                 "jerk_max", "jerk_median", "jerk_mean", "jerk_std_div", "jerk_iqr_range",

                                     "displacement_peak_magnitude","displacement_entropy", "displacement_half_point", "displacement_total_power",

                                 "displacement_power_bands_0.5_to_1","displacement_power_bands_0_to_2", "displacement_power_bands_0_to_4", "displacement_power_bands_0_to_6",

                                 "velocity_peak_magnitude","velocity_entropy", "velocity_half_point", "velocity_total_power",

                                 "velocity_power_bands_0.5_to_1","velocity_power_bands_0_to_2", "velocity_power_bands_0_to_4", "velocity_power_bands_0_to_6",

                                 "convexhull", "UDysRS_rating"])



def find_rating(record_id, sub_group):

    order = {"Neck":0,

             "Rarm":1,

             "Larm":2,

             "Trunk":3,

             "Rleg":4,

             "Lleg":5}

    try:

        rating = ratings['Communication'][str(record_id)][order[sub_group]]

    except:

        rating = 0

    return rating

    

for record_id, group in groups:

    #print(record_id)

    for index, dict_ in group.iterrows():

        position_name = dict_["position_name"]

        for sub_score, values in sub_score_dict.items():

            if position_name in values:

                #print(key, position_name)

                dict_["sub_score"] = sub_score

                dict_["UDysRS_rating"] = find_rating(dict_["record_id"], sub_score)

                #print(dict_)

                processed_df = processed_df.append(dict_, ignore_index=True)

                

    #print(group.head(17))

processed_df.head()
grouped_df = processed_df.groupby(['record_id', 'sub_score']).mean().reset_index()

grouped_df.head(7)
sub_score_gr = grouped_df.groupby(["sub_score"])

for sub_score, sub_score_group in sub_score_gr:

    print(sub_score)

    y = sub_score_group["UDysRS_rating"].astype('float64') 

    X = sub_score_group.drop(['record_id', 'sub_score', 'UDysRS_rating'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    #print(X.head())

    reg = LinearRegression()

    reg.fit(X_train, y_train)

    filename = '{}_model.sav'.format(sub_score)

    pickle.dump(reg, open(filename, 'wb'))

    print(reg.score(X_test, y_test))

    
sub_score_gr = grouped_df.groupby(["sub_score"])

for sub_score, sub_score_group in sub_score_gr:

    print(sub_score)

    y = sub_score_group["UDysRS_rating"].astype(str)

    X = sub_score_group.drop(['record_id', 'sub_score', 'UDysRS_rating'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    #print(X.head())

    clf = DecisionTreeClassifier()

    clf.fit(X_train, y_train)

    filename = '{}_decision_tree_model.sav'.format(sub_score)

    pickle.dump(reg, open(filename, 'wb'))

    print(clf.score(X_test, y_test))

    
