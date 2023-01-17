import cv2

import pandas as pd

import numpy  as np

import plotly.express    as px

import matplotlib.pyplot as plt

from sklearn.utils       import resample



from sklearn.model_selection import train_test_split

from sklearn.linear_model    import LogisticRegression

from sklearn.metrics         import roc_auc_score, accuracy_score

from IPython.display         import YouTubeVideo
YouTubeVideo('mkYBxfKDyv0', width=800, height=450)
YouTubeVideo('hXYd0WRhzN4', width=800, height=450)
# Read files

train_df = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

test_df  = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
print('Training set contains {} images of {} unique patients, resulting in a ratio of {} images per patient'.format(train_df.shape[0],

                                                                                                                     train_df.patient_id.nunique(),

                                                                                                                     round(train_df.shape[0] / train_df.patient_id.nunique(),2)

                                                                                                                    ))



print('Testing  set contains {} images of {}  unique patients, resulting in a ratio of {} images per patient'.format(test_df.shape[0],

                                                                                                                 test_df.patient_id.nunique(),

                                                                                                                 round(test_df.shape[0] / test_df.patient_id.nunique(),2)

                                                                                                                ))
# Encode 'unknowns' as NaNs

train_df['diagnosis'] = train_df.diagnosis.apply(lambda x: np.nan if x == 'unknown' else x)



labels_df = pd.DataFrame(train_df.benign_malignant.value_counts()).reset_index()

labels_df.columns = ['Label','Count']



# Create dataframe counting NaN values per column

nan_df = pd.DataFrame(train_df.isna().sum()).reset_index()

nan_df.columns  = ['Column', 'NaN_Count']

nan_df['NaN_Count'] = nan_df['NaN_Count'].astype('int')

nan_df['NaN_%'] = round(nan_df['NaN_Count']/train_df.shape[0] * 100,1)

nan_df['Type']  = 'Missingness'

nan_df.sort_values('NaN_%', inplace=True)





# Add completeness

for i in range(nan_df.shape[0]):

    complete_df = pd.DataFrame([nan_df.loc[i,'Column'],train_df.shape[0] - nan_df.loc[i,'NaN_Count'],100 - nan_df.loc[i,'NaN_%'], 'Completeness']).T

    complete_df.columns  = ['Column','NaN_Count','NaN_%','Type']

    complete_df['NaN_%'] = complete_df['NaN_%'].astype('int')

    complete_df['NaN_Count'] = complete_df['NaN_Count'].astype('int')

    nan_df = nan_df.append(complete_df, sort=True)

    

    

# Missingness Plot

fig = px.bar(nan_df,

             x='Column',

             y='NaN_%',

             title='Missingness on the Training Set',

             color='Type',

             template='plotly_dark',

             opacity = 0.6,

             color_discrete_sequence=['#dbdbdb','#38cae0'])



fig.update_xaxes(title='Column Name')

fig.update_yaxes(title='NaN Percentage')

fig.show()
# Count NaNs

train_df.isnull().sum()
labels_df = pd.DataFrame(train_df.benign_malignant.value_counts()).reset_index()

labels_df.columns = ['Label','Count']



# Create dataframe counting NaN values per column

nan_df = pd.DataFrame(test_df.isna().sum()).reset_index()

nan_df.columns  = ['Column', 'NaN_Count']

nan_df['NaN_Count'] = nan_df['NaN_Count'].astype('int')

nan_df['NaN_%'] = round(nan_df['NaN_Count']/test_df.shape[0] * 100,1)

nan_df['Type']  = 'Missingness'

nan_df.sort_values('NaN_%', inplace=True)





# Add completeness

for i in range(nan_df.shape[0]):

    complete_df = pd.DataFrame([nan_df.loc[i,'Column'],test_df.shape[0] - nan_df.loc[i,'NaN_Count'],100 - nan_df.loc[i,'NaN_%'], 'Completeness']).T

    complete_df.columns  = ['Column','NaN_Count','NaN_%','Type']

    complete_df['NaN_%'] = complete_df['NaN_%'].astype('int')

    complete_df['NaN_Count'] = complete_df['NaN_Count'].astype('int')

    nan_df = nan_df.append(complete_df, sort=True)

    

    

# Missingness Plot

fig = px.bar(nan_df,

             x='Column',

             y='NaN_%',

             title='Missingness on the Testing Set',

             color='Type',

             template='plotly_dark',

             opacity = 0.6,

             color_discrete_sequence=['#dbdbdb','#38cae0'])



fig.update_xaxes(title='Column Name')

fig.update_yaxes(title='NaN Percentage')

fig.show()
# Count NaNs

test_df.isnull().sum()
# Summarise data

count_df = labels_df.iloc[::-1]



# Create annotations

annotations = [dict(

            y=count_df.loc[i,'Label'],

            x=count_df.loc[i,'Count'] + 1000,

            text=str(round(count_df.loc[i,'Count']/train_df.shape[0]*100,1))+'%',

            font=dict(

            size=14,

            color="#000000"

            ),

            bordercolor="#c7c7c7",

            borderwidth=1,

            borderpad=4,

            bgcolor="#ffffff",

            opacity=0.95,

            showarrow=False,

        ) for i in range(count_df.shape[0])]







fig = px.bar(labels_df,

             y = 'Label',

             x = 'Count',

             title       = 'Label Distribution',

             template    = 'plotly_dark',

             orientation = 'h',

             opacity     = 0.7,

             color       = 'Label',

             color_discrete_sequence = ['#38cae0','#d1324d'] 

            )





fig.update_layout(showlegend=False, annotations = annotations)

fig.show()
fig = px.histogram(train_df,

             x     = 'age_approx',

             color = 'target',

             color_discrete_sequence = ['#38cae0','#d1324d'],

             barnorm  = 'fraction',

             template = 'plotly_dark',

             opacity  = 0.7,

             title    = 'Impact of Age across Diagnosis'

            )



fig.update_xaxes(title = 'Approximated Age', tickvals = list(range(0,91,5)))

fig.update_yaxes(title = 'Percentage of class total')

fig.show()
parallel_df = train_df.copy()



undersampled_df = pd.concat([parallel_df.query("target == 1"),resample(parallel_df.query("target == 0"),

                                                                       replace   = False,

                                                                       n_samples = 584,

                                                                       random_state = 451)

                            ],axis=0)





keep_list = ['sex','age_approx','anatom_site_general_challenge','target']

fig = px.parallel_categories(undersampled_df[keep_list],

                              color="target",

                              template='plotly_dark',

                              labels={"age_approx": "Approximate Age","sex": "Sex", 'anatom_site_general_challenge':'Anatomical Site','target':'Melanoma'},

                              color_continuous_scale=['#dbdbdb','#38cae0'],

                              title='Categorical Flow'

                             )



fig.update_layout(showlegend=False)

fig.show()
def prepare_dataframe(df):

    df['sex'] = np.where(df['sex'] == 'female',1,0)

    df = pd.concat([df.drop('anatom_site_general_challenge',axis=1), pd.get_dummies(df['anatom_site_general_challenge'])],axis=1)

    df = df.drop(['benign_malignant','image_name','patient_id','diagnosis'],axis=1)

    df.loc[df['age_approx'].isnull(),'age_approx'] = df['age_approx'].median()

    

    return(df)



def evaluate_predictions(preds, test_labels):

    '''

    Evaluate Predictions Function

    Returns accuracy and auc of the model

    '''

    auroc = roc_auc_score(test_labels.astype('uint8'), preds)

    accur = accuracy_score(test_labels.astype('uint8'), preds >= 0.5)

    print('Accuracy: ' + str(accur))

    print('AUC: ' + str(auroc))
# Split data

train, probe = train_test_split(prepare_dataframe(train_df),

                                test_size = 0.3,

                                stratify = train_df['target'],

                                random_state = 451

                               )



train_y = train.pop('target')

probe_y = probe.pop('target')
logit_model = LogisticRegression(random_state=451, solver='lbfgs', max_iter=1000)

logit_model.fit(train, train_y)



logit_preds = logit_model.predict_proba(probe)

evaluate_predictions(logit_preds[:,1], probe_y)
fig = px.bar(y = logit_model.coef_.tolist()[0],

       x = probe.columns.tolist(),

       template = 'plotly_dark',

       title = 'Logistic Regression Coefficient Values',

       color = logit_model.coef_.tolist()[0],

       color_continuous_scale = ['#d1285b','#28b5d1'],

       opacity = 0.7

      )



fig.update_yaxes(title = 'Coefficient Value')

fig.update_xaxes(title = 'Variable Name')

fig.show()
def plot_multiple_images(image_dataframe, rows = 4, columns = 4, figsize = (16, 20), resize=(1024,1024), preprocessing=None, label = 0):

    '''

    Plots Multiple Images

    Reads, resizes, applies preprocessing if desired and plots multiple images from a given dataframe

    '''

    query_string    = 'target == {}'.format(label)

    image_dataframe = image_dataframe.query(query_string).reset_index(drop=True)

    fig = plt.figure(figsize=figsize)

    ax  = []

    base_path = '../input/siim-isic-melanoma-classification/jpeg/train/'

    

    for i in range(rows * columns):

        img = plt.imread(base_path + image_dataframe.loc[i,'image_name'] + '.jpg')

        img = cv2.resize(img, resize)

        

        if preprocessing:

            img = preprocessing(img)

        

        ax.append(fig.add_subplot(rows, columns, i+1) )

        plot_title = "Image {}: {}".format(str(i+1), 'Benign' if label == 0 else 'Malignant') 

        ax[-1].set_title(plot_title)

        plt.imshow(img, alpha=1, cmap='gray')

    

    plt.show()





plot_multiple_images(train_df, label = 0)
plot_multiple_images(train_df, label = 1)