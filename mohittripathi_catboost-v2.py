import pandas as pd



import numpy as np

import matplotlib.pyplot as plt

import pandas.io.sql as psql

import seaborn as sns

from scipy import stats

import cufflinks as cf

import plotly.graph_objs as go 

from plotly.offline import init_notebook_mode,plot,iplot

from mlxtend.preprocessing import minmax_scaling

cf.go_offline()

init_notebook_mode(connected=True)

%matplotlib inline

import plotly

from plotly import tools

import plotly.tools as tls

import warnings

warnings.filterwarnings('ignore')

from wordcloud import WordCloud

from plotly.offline import  plot

import catboost as cb

import hyperopt

from catboost import Pool, CatBoostClassifier, cv

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



from sklearn.neighbors import KNeighborsClassifier



#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics













#data = pd.read_csv('train.csv')

#test = pd.read_csv('test.csv')

#state = pd.read_csv('state_labels.csv')

#print(data.head)



data = pd.read_csv("../input/train/train.csv")

test = pd.read_csv('../input/test/test.csv')

state = pd.read_csv('../input/train/train.csv')



    













### find null or NA values in data set

print(data.columns[data.isnull().any()])

#print(data.columns[data.isna().any()])

print(data.isna().sum())



adopt_speed = data.AdoptionSpeed.value_counts()



l = ('No adoption','2nd & 3rd month','1st month','1st week','same day')



adopt_speed.index = l

#print(adopt_speed)

total_pets = adopt_speed.values.sum()

#print(total_pets)

go1 = go.Bar(

            x=adopt_speed.index,

            y=(adopt_speed.values/total_pets)*100,

            marker=dict(

                color='rgb(158,202,225)',

                line=dict(

                    color='rgb(8,48,107)',

                    width=1.5),

            ),

            opacity=1

        )



dat = [go1]

layout=go.Layout(title="Adoption Speed", xaxis={'title':'Speed'}, yaxis={'title':'% of adoptions'}, width=600, height=500)

figure=go.Figure(data=dat,layout=layout)

iplot(figure)
animal_type = data.Type.value_counts()

l = ('dog','cat')

animal_type.index = l;

#print(animal_type)

total_pets = animal_type.values.sum()





go2 = go.Bar(

            x=animal_type.index,

            y=(animal_type.values/total_pets)*100,

            marker=dict(

                color='rgb(158,202,225)',

                line=dict(

                    color='rgb(8,48,107)',

                    width=1.5),

            ),

            opacity=1

        )



dat1 = [go2]

layout=go.Layout(title="Animal Types", xaxis={'title':'Type'}, yaxis={'title':'No. of animals'}, width=300, height=400)

figure=go.Figure(data=dat1,layout=layout)

iplot(figure)
# gender distribution



#print(data)

data_frame1 = data[["Gender"]]

#print(data_frame1)

data_frame_1 = data_frame1.groupby(['Gender']).size()

df4 = data_frame_1.rename({1: "Male", 2: 'Female', 3: "Mixed"})

#print(df4.index)



trace = go.Pie(labels=df4.index, values=data_frame_1.values)

dat16 = [trace]

layout=go.Layout(title="Distribution of gender of pets", xaxis={'title':'State'}, yaxis={'title':'No of Campaigns'}, width=800, height=600)

figure=go.Figure(data=dat16,layout=layout)

iplot(figure)
data_frame = data[['State','AdoptionSpeed']]

data_frame_1 = data_frame.groupby(['AdoptionSpeed','State']).size()



dist_df = data_frame_1.reset_index(level=[0,1])

data_frame2 = dist_df.rename(columns={'AdoptionSpeed': 'AdoptionSpeed', 'State': 'StateID', 0 : 'PetCount'})





df_melaka = data_frame2.loc[data_frame2['StateID'] == 41324]

lab =  df_melaka['AdoptionSpeed'].rename({0: "Same day", 12: "1 Week", 25 :'1st Month', 38: "2-3 Months", 52: "No Adoption"})





trace = go.Pie(labels=lab.index, values=df_melaka.PetCount)

dat = [trace]

layout=go.Layout(title="Distribution of Adoption Speed for State Melaka", xaxis={'title':'State'}, yaxis={'title':'No of Campaigns'}, width=500, height=300)

figure=go.Figure(data=dat,layout=layout)

iplot(figure)



df_Kedah = data_frame2.loc[data_frame2['StateID'] == 41325]



lab =  df_Kedah['AdoptionSpeed'].rename({1: "Same day", 13: "1 Week", 26 :'1st Month', 39: "2-3 Months", 53: "No Adoption"})







trace = go.Pie(labels=lab.index, values=df_Kedah.PetCount)

dat1 = [trace]

layout=go.Layout(title="Distribution of Adoption Speed for State Kedah", xaxis={'title':'State'}, yaxis={'title':'No of Campaigns'}, width=500, height=300)

figure=go.Figure(data=dat1,layout=layout)

iplot(figure)





df_Selangor = data_frame2.loc[data_frame2['StateID'] == 41326]

lab =  df_Selangor['AdoptionSpeed'].rename({2: "Same day", 14: "1 Week", 27 :'1st Month', 40: "2-3 Months", 54: "No Adoption"})







trace = go.Pie(labels=lab.index, values=df_Selangor.PetCount)

dat2 = [trace]

layout=go.Layout(title="Distribution of Adoption Speed for State Selangor", xaxis={'title':'State'}, yaxis={'title':'No of Campaigns'}, width=500, height=300)

figure=go.Figure(data=dat2,layout=layout)

iplot(figure)





df_Pulau_Pinang = data_frame2.loc[data_frame2['StateID'] == 41327]

lab =  df_Pulau_Pinang['AdoptionSpeed'].rename({3: "Same day", 15: "1 Week", 28 :'1st Month', 41: "2-3 Months", 55: "No Adoption"})







trace = go.Pie(labels=lab.index, values=df_Pulau_Pinang.PetCount)

dat3 = [trace]

layout=go.Layout(title="Distribution of Adoption Speed for State Pulau Pinang", xaxis={'title':'State'}, yaxis={'title':'No of Campaigns'}, width=500, height=300)

figure=go.Figure(data=dat3,layout=layout)

iplot(figure)



df_Perak = data_frame2.loc[data_frame2['StateID'] == 41330]

lab =  df_Perak['AdoptionSpeed'].rename({4: "Same day", 16: "1 Week", 29 :'1st Month', 42: "2-3 Months", 56: "No Adoption"})





trace = go.Pie(labels=lab.index, values=df_Perak.PetCount)

dat4 = [trace]

layout=go.Layout(title="Distribution of Adoption Speed for State Perak", xaxis={'title':'State'}, yaxis={'title':'No of Campaigns'}, width=500, height=300)

figure=go.Figure(data=dat4,layout=layout)

iplot(figure)





df_Negeri_Sembilan = data_frame2.loc[data_frame2['StateID'] == 41332]

lab =  df_Negeri_Sembilan['AdoptionSpeed'].rename({5: "Same day", 17: "1 Week", 30 :'1st Month', 43: "2-3 Months", 57: "No Adoption"})





trace = go.Pie(labels=lab.index, values=df_Negeri_Sembilan.PetCount)

dat5 = [trace]

layout=go.Layout(title="Distribution of Adoption Speed for State Negeri Sembilan", xaxis={'title':'State'}, yaxis={'title':'No of Campaigns'}, width=500, height=300)

figure=go.Figure(data=dat5,layout=layout)

iplot(figure)





df_Pahang = data_frame2.loc[data_frame2['StateID'] == 41335]

lab =  df_Pahang['AdoptionSpeed'].rename({6: "Same day", 18: "1 Week", 31 :'1st Month', 4: "2-3 Months", 58: "No Adoption"})







trace = go.Pie(labels=lab.index, values=df_Pahang.PetCount)

dat6 = [trace]

layout=go.Layout(title="Distribution of Adoption Speed for State Pahang", xaxis={'title':'State'}, yaxis={'title':'No of Campaigns'}, width=500, height=300)

figure=go.Figure(data=dat6,layout=layout)

iplot(figure)



df_Johor = data_frame2.loc[data_frame2['StateID'] == 41336]





trace = go.Pie(labels=df_Johor.AdoptionSpeed, values=df_Johor.PetCount)

dat7 = [trace]

layout=go.Layout(title="Distribution of Adoption Speed for State Johor", xaxis={'title':'State'}, yaxis={'title':'No of Campaigns'}, width=500, height=300)

figure=go.Figure(data=dat7,layout=layout)

iplot(figure)







df_Sarawak = data_frame2.loc[data_frame2['StateID'] == 41342]





trace = go.Pie(labels=df_Sarawak.AdoptionSpeed, values=df_Sarawak.PetCount)

dat8 = [trace]

layout=go.Layout(title="Distribution of Adoption Speed for State Sarawak", xaxis={'title':'State'}, yaxis={'title':'No of Campaigns'}, width=500, height=300)

figure=go.Figure(data=dat8,layout=layout)

iplot(figure)







df_Sabah = data_frame2.loc[data_frame2['StateID'] == 41345]





trace = go.Pie(labels=df_Sabah.AdoptionSpeed, values=df_Sabah.PetCount)

dat9 = [trace]

layout=go.Layout(title="Distribution of Adoption Speed for State Sabah", xaxis={'title':'State'}, yaxis={'title':'No of Campaigns'}, width=500, height=300)

figure=go.Figure(data=dat9,layout=layout)

iplot(figure)









df_Kelantan = data_frame2.loc[data_frame2['StateID'] == 41367]





trace = go.Pie(labels=df_Kelantan.AdoptionSpeed, values=df_Kelantan.PetCount)

dat10 = [trace]

layout=go.Layout(title="Distribution of Adoption Speed for State Kelantan", xaxis={'title':'State'}, yaxis={'title':'No of Campaigns'}, width=500, height=300)

figure=go.Figure(data=dat10,layout=layout)

iplot(figure)









df_Kuala_Lumpur = data_frame2.loc[data_frame2['StateID'] == 41401]





trace = go.Pie(labels=df_Kuala_Lumpur.AdoptionSpeed, values=df_Kuala_Lumpur.PetCount)

dat11 = [trace]

layout=go.Layout(title="Distribution of Adoption Speed for State Kuala Lumpur", xaxis={'title':'State'}, yaxis={'title':'No of Campaigns'}, width=500, height=300)

figure=go.Figure(data=dat11,layout=layout)

iplot(figure)





df_Terengganu = data_frame2.loc[data_frame2['StateID'] == 41361]





trace = go.Pie(labels=df_Terengganu.AdoptionSpeed, values=df_Terengganu.PetCount)

dat12 = [trace]

layout=go.Layout(title="Distribution of Adoption Speed for State Terengganu", xaxis={'title':'State'}, yaxis={'title':'No of Campaigns'}, width=500, height=300)

figure=go.Figure(data=dat12,layout=layout)

iplot(figure)





df_Labuan = data_frame2.loc[data_frame2['StateID'] == 41415]





trace = go.Pie(labels=df_Labuan.AdoptionSpeed, values=df_Labuan.PetCount)

dat13 = [trace]

layout=go.Layout(title="Distribution of Adoption Speed for State Labuan", xaxis={'title':'State'}, yaxis={'title':'No of Campaigns'}, width=500, height=300)

figure=go.Figure(data=dat13,layout=layout)

iplot(figure)
# age wise pets distribution for age < 120 months

data1 = data[data['Age'] < 120]

#print(data1)

age = data1.Age.value_counts()





#print(age)



go2 = go.Bar(

            x=age.index,

            y=age.values,

            marker=dict(

                color='rgb(158,202,225)',

                line=dict(

                    color='rgb(8,48,107)',

                    width=1.5),

            ),

            opacity=1

        )



dat17 = [go2]

layout=go.Layout(title="Age distribution", xaxis={'title':'Age in Months'}, yaxis={'title':'Frequency'}, width=1100, height=600)

figure=go.Figure(data=dat17,layout=layout)

iplot(figure)



def show_wordcloud(data, title = None):

    '''Split names by space and generate word counts.'''

    wordcloud = WordCloud(

        background_color='white',

        max_words=100,

        max_font_size=40,

        scale=3,

        random_state=1 # chosen at random by flipping a coin it was heads

    ).generate(str(data))



    fig = plt.figure(1, figsize=(12, 12))

    plt.axis('off')

    if title:

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()

    

    

# adopted dogs

show_wordcloud(data[data.Type == 1]['Name'])
# adopted cats

show_wordcloud(data[data.Type == 2]['Name'])
missing_names_data = data[data["Name"].isnull()].groupby("AdoptionSpeed").size()

total_missing_names = missing_names_data.values.sum()

#print(missing_names_data)

#print(missing_names_data.values.sum())

with_names_data = data[-data["Name"].isnull()].groupby("AdoptionSpeed").size()

total_with_names = with_names_data.values.sum()

#print(with_names_data)

#print(with_names_data.values.sum())



l = ('same day','1st week','1st month','2nd & 3rd month','No adoption')

missing_names_data.index = l;

with_names_data.index = l;



go5 = go.Bar(

            x=missing_names_data.index,

            y=(missing_names_data.values/total_missing_names) * 100,

            marker=dict(

                color='rgb(158,202,225)',

                line=dict(

                    color='rgb(8,48,107)',

                    width=1.5),

            ),

            opacity=1,name='missing name'

        )



go6 = go.Bar(

            x=with_names_data.index,

            y=(with_names_data.values/total_with_names)*100,

            marker=dict(

                color='brown',

                line=dict(

                    color='rgb(8,48,107)',

                    width=1.5),

            ),

            opacity=1,name='with Names'

        )





dat18 = [go5,go6]

layout=go.Layout(title="Adoption speed comparison for pets with missing and with names", xaxis={'title':'Adoption Speed'}, yaxis={'title':'% Pets Adopted'}, width=900, height=600)

figure=go.Figure(data=dat18,layout=layout)

iplot(figure)
l = ('same day','1st week','1st month','2nd & 3rd month','No adoption')

## adoption speed of steralized pets

data2 = data[data['Sterilized'] == 1]

#print(data2)

#print(data_frame1)

data_frame_str = data2.groupby(['AdoptionSpeed']).size()

total_str = data_frame_str.values.sum()

#print(data_frame_str)

data_frame_str.index = l





## adoption speed of non steralized pets

data3 = data[data['Sterilized'] == 2]

#print(data2)

#print(data_frame1)

data_frame_notstr = data3.groupby(['AdoptionSpeed']).size()

total_nonstr = data_frame_notstr.values.sum()

#print(data_frame_notstr)

data_frame_notstr.index = l





## adoption speed of pets with no info about steralization

data4 = data[data['Sterilized'] == 3]

#print(data2)

#print(data_frame1)

data_frame_noinfo = data4.groupby(['AdoptionSpeed']).size()

total_noinfo = data_frame_noinfo.values.sum()

#print(data_frame_noinfo)

data_frame_noinfo.index = l







go2 = go.Bar(

            x=data_frame_str.index,

            y=(data_frame_str.values/total_str)*100,

            marker=dict(

                color='rgb(158,202,225)',

                line=dict(

                    color='rgb(8,48,107)',

                    width=1.5),

            ),

            opacity=1,name='Steralized'

        )



go3 = go.Bar(

            x=data_frame_notstr.index,

            y=(data_frame_notstr.values/total_nonstr)*100,

            marker=dict(

                color='brown',

                line=dict(

                    color='rgb(8,48,107)',

                    width=1.5),

            ),

            opacity=1,name='Non Steralized'

        )







go4 = go.Bar(

            x=data_frame_noinfo.index,

            y=(data_frame_noinfo.values/total_noinfo)*100,

            marker=dict(

                color='darkblue',

                line=dict(

                    color='rgb(8,48,107)',

                    width=1.5),

            ),

            opacity=1, name='No information'

        )



dat17 = [go2,go3,go4]

layout=go.Layout(title="Adoption speed for steralized,non steralized and pets with no info about steralization", xaxis={'title':'Adoption Speed'}, yaxis={'title':'% Pets Adopted'}, width=900, height=600)

figure=go.Figure(data=dat17,layout=layout)

iplot(figure)
##### Modelling

data_new = data.drop(data.columns[[1,18,20,21]], axis=1)

print(data_new.dtypes)

test_new = test.drop(data.columns[[1,18,20,21]], axis=1)

print(test_new.dtypes)





from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import ParameterGrid

from sklearn.model_selection import train_test_split

from itertools import product, chain

from tqdm import tqdm
#test_new = test.drop(data.columns[[1,18,20,21]], axis=1)

RANDOM_STATE = 0



def get_x(df):

    df = df.drop(data.columns[[1,18,20,21]], axis=1)

    

    columns = list(df.columns)

    if 'AdoptionSpeed' in columns:

        columns.remove('AdoptionSpeed')

    cat_features = np.where(df[columns].dtypes != np.float)[0]

    return df[columns].values, cat_features





def get_xy(df):

    X, _ = get_x(df)

    y = df['AdoptionSpeed']

    return X, y



#  

def cross_val(X, y, X_test, param, cat_features, n_splits=3):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    

    acc = []

    predict = None

    

    for tr_ind, val_ind in skf.split(X, y):

        X_train = X[tr_ind]

        y_train = y[tr_ind]

        

        X_valid = X[val_ind]

        y_valid = y[val_ind]

        

        clf = CatBoostClassifier(iterations=500,

                                loss_function = param['loss_function'],

                                depth=param['depth'],

                                l2_leaf_reg = param['l2_leaf_reg'],

                                eval_metric = 'Accuracy',

                                leaf_estimation_iterations = 10,

                                use_best_model=True,

                                logging_level='Silent'

        )

        

        clf.fit(X_train, 

                y_train,

                cat_features=cat_features,

                eval_set=(X_valid, y_valid)

        )

        

        y_pred = clf.predict(X_valid)

        accuracy = accuracy_score(y_valid, y_pred)

        acc.append(accuracy)

    return sum(acc)/n_splits

    

def catboost_GridSearchCV(X, y, X_test, params, cat_features, n_splits=5):

    ps = {'acc':0,

          'param': []

    }

    

    predict=None

    

    for prms in tqdm(list(ParameterGrid(params)), ascii=True, desc='Params Tuning:'):

                          

        acc = cross_val(X, y, X_test, prms, cat_features, n_splits=5)



        if acc>ps['acc']:

            ps['acc'] = acc

            ps['param'] = prms

    print('Acc: '+str(ps['acc']))

    print('Params: '+str(ps['param']))

    

    return ps['param']

    

    

def main():

    train = pd.read_csv("../input/train/train.csv")

    test = pd.read_csv('../input/test/test.csv')

    

    

    X_train, y_train = get_xy(train)

    X_test, cat_features = get_x(test)

    

    params = {'depth':[2, 3, 4],

              'loss_function': ['MultiClass'],

              'l2_leaf_reg':np.logspace(-20, -19, 3)

    }

    

    param = catboost_GridSearchCV(X_train, y_train, X_test, params, cat_features)



    clf = CatBoostClassifier(iterations=2500,

                            loss_function = param['loss_function'],

                            depth=param['depth'],

                            l2_leaf_reg = param['l2_leaf_reg'],

                            eval_metric = 'Accuracy',

                            leaf_estimation_iterations = 10,

                            use_best_model=True

    )

    X_train, X_valid, y_train, y_valid = train_test_split(X_train,

                                                        y_train, 

                                                        shuffle=True,

                                                        random_state=RANDOM_STATE,

                                                        train_size=0.8,

                                                        stratify=y_train

    )

    clf.fit(X_train, 

            y_train,

            cat_features=cat_features,

            logging_level='Silent',

            eval_set=(X_valid, y_valid)

    )

    

    #print(clf.predict(X_test))

    sub = pd.DataFrame({'PetID': test['PetID'],'AdoptionSpeed':[int(i) for i in clf.predict(X_test)]})

    sub.to_csv('submission.csv',index=False)    



    

if __name__=='__main__':

    main()