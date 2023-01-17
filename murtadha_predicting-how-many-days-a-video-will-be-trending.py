from IPython.display import Image

Image("/kaggle/input/project-figures/figure1.PNG")
Image("/kaggle/input/project-figures/figure2.PNG")
Image("/kaggle/input/project-figures/figure3.PNG")
Image("/kaggle/input/project-figures/table1.PNG")
Image("/kaggle/input/project-figures/table2.PNG")
Image("/kaggle/input/project-figures/figure6.PNG")
Image("/kaggle/input/project-figures/figure7.PNG")
Image("/kaggle/input/project-figures/figure8.PNG")
Image("/kaggle/input/project-figures/figure9.PNG")
#This script is to read the excel files and create a ndarray of length n video and each cell in this array contains

# multiple feature vectors and each of them represents one day while trending



#Features: how many days the video was trending (including same day recording), title sentiment (polarity), title sentiment (subjectivity)

# ,title average tf-idf, tags sentiment (polarity), tags sentiment (subjectivity), tags average tf-idf,

#  description sentiment (polarity), description sentiment (subjectivity), description average tf-idf, category_id,

# The time since the video was uploaded, views, likes, dislikes, comments count, like rate, dislike rate, comment rate, change in views since the trending day

#, change in likes since the trending day, change in dislikes since the trending day, change in comments count since the trending day

#, comments disabled (0 or 1), ratings disabled (0 or 1).

# 'title_tfidf', 'tags_tfidf' and 'description_tfidf' are checking if rare and significant words been used.

# Rate features are calculated to find if the video is interesting in each day

# change features are calculated to check if the number of likes, dislikes and comments are increasing or decreasing in each day.

#TODO add more features



import pandas as pd

import numpy as np

from textblob import TextBlob

import string,math,pickle



def remove_punctuation (doc):

    return " ".join("".join([" " if ch in string.punctuation else ch for ch in doc]).split())



def tfidf_fun(word, blob, bloblist):

    tf=blob.words.count(word) / len(blob.words)

    n_containing=sum(1 for blob2 in bloblist if word in blob2.words)

    idf=math.log(len(bloblist) / (1 + n_containing))

    return tf * idf



def avg_tfidf_fun(blob, bloblist):

    avg_tfidf=0

    for wordI in blob.words:

        avg_tfidf=avg_tfidf+tfidf_fun(wordI, blob, bloblist)

    if blob.words.__len__() != 0:

        avg_tfidf=avg_tfidf/blob.words.__len__()

        return avg_tfidf

    else:

        return 0



################################## Feature Extraction ###############################################

numFeatures=25

df_yout = pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")



#create the features and labels dataframe

df_features = pd.DataFrame(np.zeros([df_yout.shape[0],numFeatures]),columns=['days_was_trending', 'title_polarity', 'title_subjectivity','title_tfidf' ,

                                                                             'tags_polarity', 'tags_subjectivity' , 'tags_tfidf','description_polarity',

                                                                             'description_subjectivity', 'description_tfidf',

                                                                             'category_id','days_since_uploaded','views',

                                                                             'likes','dislikes','comment_count','like_rate','dislike_rate',

                                                                             'comment_rate','view_change','like_change','dislike_change','comment_change',

                                                                             'comments_disabled','ratings_disabled'])

                                    # 'title_tfidf', 'tags_tfidf' and 'description_tfidf' take too much time to be calculated. Checking if rare and significant words been used.



df_label = pd.DataFrame(np.zeros([df_yout.shape[0],2]),columns=['video_ind','days_will_be_trending'])



#construct bloblist by combining all the titles, tags and descriptions into multiple documents

unique_ids=df_yout['video_id'].unique()

bloblist=[]

for docI in range(unique_ids.shape[0]):

    vedI_data = df_yout[df_yout['video_id'] == unique_ids[docI]]

    vedI_data=vedI_data.reset_index(drop=True)

    document = vedI_data.title[0]+ " " +vedI_data.tags[0]+" " +str(vedI_data.description[0])

    bloblist.append(TextBlob(remove_punctuation(document)))



#Calculating the features for each vedio for each day

count1=0

for vidI in range(100):#range(unique_ids.shape[0]): #TODO remove the comment to use all the videos and not just 10

    print('Extracting the features of video #:' + str(vidI))

    vedI_data=df_yout[df_yout['video_id']==unique_ids[vidI]]

    vedI_data=vedI_data.reset_index(drop=True)

    total_num_trending=vedI_data.shape[0]



    titleI = TextBlob(remove_punctuation(vedI_data.title[0])) #removing the punctuation results in changing sentiment results slightly

    tagsI = TextBlob(remove_punctuation(vedI_data.tags[0]))

    descriptionI = TextBlob(remove_punctuation(str(vedI_data.description[0])))

    title_tfidf=avg_tfidf_fun(titleI,bloblist)

    tags_tfidf = avg_tfidf_fun(tagsI, bloblist)

    description_tfidf = avg_tfidf_fun(descriptionI, bloblist)



    trending_date=pd.to_datetime(vedI_data.trending_date[0], format='%y.%d.%m')

    published_date=pd.to_datetime(vedI_data.publish_time[0], format='%Y-%m-%dT%H:%M:%S.%fZ')

    date_published_trending=trending_date -published_date

    days_published_trending=float(date_published_trending._d)+(float(date_published_trending._h)/24)



    for dayI in range(total_num_trending):

        #print('day number :' + str(dayI))

        df_label.video_ind[count1] = vidI

        df_label.days_will_be_trending[count1]=total_num_trending-dayI-1



        df_features.days_was_trending[count1]=dayI+1



        df_features.title_polarity[count1] =titleI.sentiment.polarity

        df_features.title_subjectivity[count1] = titleI.sentiment.subjectivity

        df_features.title_tfidf[count1]=title_tfidf



        df_features.tags_polarity[count1] =tagsI.sentiment.polarity

        df_features.tags_subjectivity[count1] = tagsI.sentiment.subjectivity

        df_features.tags_tfidf[count1]=tags_tfidf



        df_features.description_polarity[count1] =descriptionI.sentiment.polarity #average for all the sentences

        df_features.description_subjectivity[count1] = descriptionI.sentiment.subjectivity

        df_features.description_tfidf[count1]=description_tfidf



        df_features.category_id[count1]=vedI_data.category_id[dayI]

        df_features.days_since_uploaded[count1]=days_published_trending+dayI+1

        df_features.views[count1]=vedI_data.views[dayI]

        df_features.likes[count1]=vedI_data.likes[dayI]

        df_features.dislikes[count1]=vedI_data.dislikes[dayI]

        df_features.comment_count[count1]=vedI_data.comment_count[dayI]



        df_features.like_rate[count1]=vedI_data.likes[dayI]/vedI_data.views[dayI]

        df_features.dislike_rate[count1]=vedI_data.dislikes[dayI]/vedI_data.views[dayI]

        df_features.comment_rate[count1]=vedI_data.comment_count[dayI]/vedI_data.views[dayI]



        if dayI==0: #no change for day 1 but give it 1 for each feature ;)

            df_features.view_change[count1]=1

            df_features.like_change[count1]=1

            df_features.dislike_change[count1]=1

            df_features.comment_change[count1]=1

        else:

            df_features.view_change[count1]=vedI_data.views[dayI]-vedI_data.views[dayI-1]

            df_features.like_change[count1]=vedI_data.likes[dayI]-vedI_data.likes[dayI-1]

            df_features.dislike_change[count1]=vedI_data.dislikes[dayI]-vedI_data.dislikes[dayI-1]

            df_features.comment_change[count1]=vedI_data.comment_count[dayI]-vedI_data.comment_count[dayI-1]



        df_features.comments_disabled[count1] = np.double(vedI_data.comments_disabled[dayI])

        df_features.ratings_disabled[count1] = np.double(vedI_data.ratings_disabled[dayI])

        count1+=1



with open('/kaggle/working/NLP_data.pkl', 'wb') as Data:

    pickle.dump([df_features, df_label, numFeatures, bloblist, unique_ids], Data)
from __future__ import print_function, division

import numpy as np, pickle, os

import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scipy import stats

from sklearn.model_selection import train_test_split

import xgboost as xgb

from imblearn.over_sampling import SMOTE

from xgboost import plot_importance



########################### Plotting functions and others #############################################################

def plot_fun(loss_train, loss_val, accuracy_train_All, accuracy_val_All):

    plt.cla()

    plt.subplot(1, 2, 1)

    plt.plot(np.arange(1, loss_train.__len__() + 1), loss_train, 'b', np.arange(1, loss_val.__len__() + 1),

             loss_val, 'g')

    plt.ylabel('Loss')

    plt.xlabel('Epoch number')

    plt.legend(['Training Loss', 'Validation Loss'])

    # plt.ylim([0, 3])



    plt.subplot(1, 2, 2)

    plt.plot(np.arange(1, accuracy_train_All.__len__() + 1), accuracy_train_All, 'b-s',

             np.arange(1, accuracy_train_All.__len__() + 1),

             accuracy_val_All, 'g-^')

    plt.ylabel('corrcoef')

    plt.xlabel('Epoch number')

    plt.legend(['Training corrcoef', 'Validation corrcoef'])

    # plt.ylim([0,100])

    plt.draw()

    plt.pause(0.000000001)



def plot_dis(x):

    num_bins = range(int(min(x)),int(max(x)+2))

    # the histogram of the data

    n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='blue', alpha=0.5)



    # add a 'best fit' line

    #y = mlab.normpdf(bins, np.mean(x), np.std(x))

    #plt.plot(bins, y, 'r--')

    plt.xlabel('Days')

    plt.ylabel('Probability')

    plt.title(r'Training Data Distribution')



    # Tweak spacing to prevent clipping of ylabel

    plt.subplots_adjust(left=0.15)

    plt.show()



def plot_feature_imp(model):

    plot_importance(model)

################################### XGBoost hyperparameters #################################################



SaveDirParent = '/kaggle/working/'

num_classes = 1

learning_rateI=0.1  #step size shrinkage used to prevent overfitting. Range is [0,1]

n_estimatorsRange=[30]#range(10,100,20) #[10,30,50,..]  number of trees you want to build.

max_depthRange=[5]#range(3,10,2) # determines how deeply each tree is allowed to grow during any boosting round.

colsample_byteeRange=[0.3]#[i/10.0 for i in range(1,5)] #percentage of features used per tree. High value can lead to overfitting.

#subsampleI=0.8 #percentage of samples used per tree. Low value can lead to underfitting.

gamaI=[0.3]#[i/10.0 for i in range(0,6)] #controls whether a given node will split based on the expected reduction

                                        # in loss after the split. A higher value leads to fewer splits. Supported only for tree-based learners.

#reg_alphaRange=[1e-5, 1e-2, 0.1, 1, 100] #alpha: L1 regularization on leaf weights. A large value leads to more regularization.



#Example https://www.datacamp.com/community/tutorials/xgboost-in-python#what

#Tunning https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/





num_classes = 1

# truncated_backprop_length = ? # not used

RandomStart = 1 / 4  # pick a random start for each signal between 0 and (RandomStart*signal length)



################################ Import the dataset #########################################

exists = os.path.isfile('/kaggle/working/NLP_data_reformed.pkl')

if exists:

    with open('/kaggle/working/NLP_data_reformed.pkl', 'rb') as Data:

        [array_features_train, array_features_val, array_features_test, array_label_train,

         array_label_val,

         array_label_test, num_samples_train, num_samples_val, num_samples_test, features_mean, features_std,

         numFeatures, bloblist, unique_ids]= pickle.load(Data)

else: # Load and reformat the split data data

    exists = os.path.isfile('/kaggle/working/NLP_data_splits.pkl')

    if exists:

        with open('/kaggle/working/NLP_data_splits.pkl', 'rb') as Data:

            [list_features_train, list_features_val, list_features_test, list_features_train, list_label_train,

             list_label_val,

             list_label_test, num_videos_train, num_videos_val, num_videos_test, features_mean, features_std,

             numFeatures,

             bloblist, unique_ids] = pickle.load(Data)

    else:  # Load and split the original data

        with open('/kaggle/working/NLP_data.pkl', 'rb') as Data:

            [df_features, df_label, numFeatures, bloblist, unique_ids] = pickle.load(Data)



        # reformat the data

        list_videos = np.array(df_label['video_ind'])  # contains indeces for all the videos with repetition.

        num_videos = np.int32(list_videos[-1])

        list_features = np.empty((num_videos,), dtype=object)

        list_label = np.empty((num_videos,), dtype=object)

        for i in range(num_videos):

            list_features[i] = df_features[df_label['video_ind'] == i]

            list_label[i] = df_label.loc[df_label['video_ind'] == i, 'days_will_be_trending']



        # Splitting the data into (60% training + 20% validation), 20%testing

        list_features_train, list_features_test, list_label_train, list_label_test = train_test_split(list_features,

                                                                                                      list_label,

                                                                                                      test_size=0.2,

                                                                                                      random_state=1)



        list_features_train, list_features_val, list_label_train, list_label_val = train_test_split(list_features_train,

                                                                                                    list_label_train,

                                                                                                    test_size=0.2,

                                                                                                    random_state=1)



        num_videos_train = list_features_train.shape[0]  # number of videos for training

        num_videos_val = list_features_val.shape[0]  # number of videos for validation

        num_videos_test = list_features_val.shape[0]  # number of videos for testing



        # normalizing the features based on mean and std of training data

        num_samples = 0

        features_mean = np.zeros(numFeatures)

        features_std = np.zeros(numFeatures)

        for video_i in range(num_videos_train):

            features_mean = np.sum(list_features_train[video_i], axis=0) + features_mean

            features_std = np.std(list_features_train[video_i], axis=0) + features_std

            num_samples = num_samples + list_features_train[video_i].shape[0]



        features_mean = features_mean / num_samples

        features_std = features_std / num_samples



        # normalize training data

        for video_i in range(num_videos_train):

            list_features_train[video_i] = (list_features_train[video_i] - features_mean) / features_std



        # normalize validation data

        for video_i in range(num_videos_val):

            list_features_val[video_i] = (list_features_val[video_i] - features_mean) / features_std

        # normalize testing data

        for video_i in range(num_videos_test):

            list_features_test[video_i] = (list_features_test[video_i] - features_mean) / features_std



        with open('/kaggle/working/NLP_data_splits.pkl', 'wb') as Data:

            pickle.dump(

                [list_features_train, list_features_val, list_features_test, list_features_train, list_label_train,

                 list_label_val,

                 list_label_test, num_videos_train, num_videos_val, num_videos_test, features_mean, features_std,

                 numFeatures,

                 bloblist, unique_ids], Data)



    # reformat the data to one array for training , validation testing

    num_samples_train = 0

    for video_i in range(num_videos_train):

        num_samples_train = num_samples_train + list_features_train[video_i].shape[0]

    num_samples_val = 0

    for video_i in range(num_videos_val):

        num_samples_val = num_samples_val + list_features_val[video_i].shape[0]

    num_samples_test = 0

    for video_i in range(num_videos_test):

        num_samples_test = num_samples_test + list_features_test[video_i].shape[0]



    array_features_train = np.zeros([num_samples_train, numFeatures])

    array_label_train = np.zeros([num_samples_train])

    countI = 0

    for video_i in range(num_videos_train):

        indFrom = countI

        indTo = indFrom + list_features_train[video_i].shape[0]

        array_features_train[indFrom:indTo, :] = np.array(list_features_train[video_i])

        array_label_train[indFrom:indTo] = np.array(list_label_train[video_i])

        countI = indTo



    array_features_val = np.zeros([num_samples_val, numFeatures])

    array_label_val = np.zeros([num_samples_val])

    countI = 0

    for video_i in range(num_videos_val):

        indFrom = countI

        indTo = indFrom + list_features_val[video_i].shape[0]

        array_features_val[indFrom:indTo, :] = np.array(list_features_val[video_i])

        array_label_val[indFrom:indTo] = np.array(list_label_val[video_i])

        countI = indTo



    array_features_test = np.zeros([num_samples_test, numFeatures])

    array_label_test = np.zeros([num_samples_test])

    countI = 0

    for video_i in range(num_videos_test):

        indFrom = countI

        indTo = indFrom + list_features_test[video_i].shape[0]

        array_features_test[indFrom:indTo, :] = np.array(list_features_test[video_i])

        array_label_test[indFrom:indTo] = np.array(list_label_test[video_i])

        countI = indTo



    with open('/kaggle/working/NLP_data_reformed.pkl', 'wb') as Data:

        pickle.dump(

            [array_features_train, array_features_val, array_features_test, array_label_train,

             array_label_val,

             array_label_test, num_samples_train, num_samples_val, num_samples_test, features_mean, features_std,

             numFeatures, bloblist, unique_ids], Data)



#Random oversampling

ros = SMOTE(random_state=42,k_neighbors=3)

array_features_train, array_label_train = ros.fit_resample(array_features_train, array_label_train)





# Saving results parameters

savingTime = 1  # save after N epochs

Results_train_All = np.zeros((len(n_estimatorsRange) * len(max_depthRange) * len(colsample_byteeRange),

                              5))  # saving the results for each fold in CV, for 160 trials and for 5 metrics

Results_val_All = np.zeros((len(n_estimatorsRange) * len(max_depthRange) * len(colsample_byteeRange), 5))  #

CVpredictionsMaxCorr = np.zeros((num_samples_test))

CVpredictionsMinRMSE = np.zeros((num_samples_test))

countSave = 0

highestInnCorr = -1  # initial values

lowestInnRMSE = 100



for n_estimatorsI in  n_estimatorsRange:

    for max_depthI in max_depthRange:

        for colsample_byteeI in colsample_byteeRange:

            xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=colsample_byteeI,

                                      learning_rate=learning_rateI,

                                      max_depth=max_depthI, gama=gamaI, n_estimators=n_estimatorsI)

            xg_reg.fit(array_features_train, array_label_train)



            # Results on training data

            predictionsPerDay = xg_reg.predict(array_features_train)



            Results_train_All[countSave, 0] = np.sqrt(mean_squared_error(array_label_train, predictionsPerDay))

            Results_train_All[countSave, 1] = mean_absolute_error(array_label_train, predictionsPerDay)

            Results_train_All[countSave, 2] = r2_score(array_label_train, predictionsPerDay)

            corrI = stats.pearsonr(array_label_train, predictionsPerDay)

            Results_train_All[countSave, 3] = corrI[0]

            Results_train_All[countSave, 4] = corrI[1]



            # Results on validation data

            predictionsPerDay = xg_reg.predict(array_features_val)



            Results_val_All[countSave, 0] = np.sqrt(mean_squared_error(array_label_val, predictionsPerDay))

            Results_val_All[countSave, 1] = mean_absolute_error(array_label_val, predictionsPerDay)

            Results_val_All[countSave, 2] = r2_score(array_label_val, predictionsPerDay)

            corrI = stats.pearsonr(array_label_val, predictionsPerDay)

            Results_val_All[countSave, 3] = corrI[0]

            Results_val_All[countSave, 4] = corrI[1]

            print("XGboost with estimators # %d, max depth # %d, training RMSE %.2f, validation RMSE %.2f" % (

            n_estimatorsI, max_depthI,

            Results_train_All[

                countSave, 0],

            Results_val_All[

                countSave, 0]))



            # Saving the model with the highest validation correlation



            if (Results_val_All[countSave, 3] > highestInnCorr and Results_val_All[countSave, 3] ==

                    Results_val_All[countSave, 3]):

                save_path = SaveDirParent + 'xgbMode' + '_CorrCV.dat'

                pickle.dump(xg_reg, open(save_path, "wb"))

                highestInnCorr = Results_val_All[countSave, 3]

                print("Model saved in file: %s" % save_path)

            if (Results_val_All[countSave, 0] < lowestInnRMSE and Results_val_All[countSave, 0] ==

                    Results_val_All[countSave, 0]):

                save_path = SaveDirParent + 'xgbMode' + '_RMSECV.dat'

                pickle.dump(xg_reg, open(save_path, "wb"))

                lowestInnRMSE = Results_val_All[countSave, 0]

                print("Model saved in file: %s" % save_path)



            countSave = countSave + 1









# Results on Testing data

# Using saved best model with highest correlation to find testing predictions and save it

save_path = SaveDirParent +'xgbMode'+ '_CorrCV.dat'

xg_reg = pickle.load(open(save_path, "rb"))

predictions_MaxCorr= xg_reg.predict(array_features_test)



# Using saved best model with lowest RMSE to find testing predictions and save it

save_path = SaveDirParent + 'xgbMode' + '_RMSECV.dat'

xg_reg = pickle.load(open(save_path, "rb"))

predictions_MinRMSE = xg_reg.predict(array_features_test)



# Finding Final Testing Results

corrI = stats.pearsonr(array_label_test, predictions_MaxCorr)

class_reportMaxCorr = [np.sqrt(mean_squared_error(array_label_test, predictions_MaxCorr)),

                       mean_absolute_error(array_label_test, predictions_MaxCorr),

                       r2_score(array_label_test, predictions_MaxCorr), corrI[0], corrI[1]]

print(

    "Final testing using Max validation Corr: RMSE %.2f, MAE %.2f, R2 score %.2f, Correlation coefficient %.2f (p=%.4f)." % (

        np.sqrt(mean_squared_error(array_label_test, predictions_MaxCorr)),

        mean_absolute_error(array_label_test, predictions_MaxCorr),

        r2_score(array_label_test, predictions_MaxCorr),

        corrI[0], corrI[1]))

corrI = stats.pearsonr(array_label_test, predictions_MinRMSE)

class_reportMinRMSE = [np.sqrt(mean_squared_error(array_label_test, predictions_MinRMSE)),

                       mean_absolute_error(array_label_test, predictions_MinRMSE),

                       r2_score(array_label_test, predictions_MinRMSE), corrI[0], corrI[1]]



print(

    "Final testing using Min validation RMSE: RMSE %.2f, MAE %.2f, R2 score %.2f, Correlation coefficient %.2f (p=%.4f)." % (

        np.sqrt(mean_squared_error(array_label_test, predictions_MinRMSE)),

        mean_absolute_error(array_label_test, predictions_MinRMSE),

        r2_score(array_label_test, predictions_MinRMSE),

        corrI[0], corrI[1]))



###############Saving the variables#################

# Saving the objects:

with open(SaveDirParent + 'XGboostResults.pkl', 'wb') as Results:  # Python 3: open(..., 'wb')

    pickle.dump(

        [Results_train_All, Results_val_All, class_reportMaxCorr, class_reportMinRMSE,

         predictions_MinRMSE, array_label_test, predictions_MaxCorr], Results)