#First import panads

import pandas as pd
#Defining function KYD, feel free to make changes in codes as per your project requirements.

def KYD(DF):

    index = list(DF.columns)

    new_df = pd.DataFrame(index=index)

    for ix in DF.columns:

        new_df.at[ix,'data_types'] = DF[ix].dtypes

        new_df.at[ix,'null_counts'] = DF[ix].isnull().sum()

        new_df.at[ix,'unique_values'] = DF[ix].nunique()

        if DF[ix].dtypes != 'object':

#Below codes are commented for a better presentation, you can uncomment as per your need.

            new_df.at[ix,'mean_value'] = DF[ix].mean()

            #new_df.at[ix,'std_value'] = DF[ix].std()

            #new_df.at[ix,'min_value'] = DF[ix].min()

            #new_df.at[ix, '25th_percentile'] = DF[ix].quantile(0.25)

            #new_df.at[ix, '50th_percentile'] = DF[ix].quantile(0.5)

            #new_df.at[ix, '75th_percentile'] = DF[ix].quantile(0.75)

            #new_df.at[ix,'max_value'] = DF[ix].max()

    print('Total Rows:', len(DF.index))

    print('Total Columns:',len(DF.columns))

    print(new_df.to_string())
#DataSet of Heart Disease UCI

heart_data = pd.read_csv('../input/heart-disease-uci/heart.csv')

KYD(heart_data)
#DataSet of Graduate Admissions

admission_data = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')

KYD(admission_data)
#DataSets of Google Play Store Apps

google_play_store_user_data = pd.read_csv('../input/google-play-store-apps/googleplaystore_user_reviews.csv')

google_play_store_data = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')

KYD(google_play_store_user_data)

print("\n")

KYD(google_play_store_data)