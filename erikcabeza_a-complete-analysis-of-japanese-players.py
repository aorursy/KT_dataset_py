import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

 















import numpy as np 

import pandas as pd 

import seaborn as sns

import plotly.express as px

fifaDataset=pd.read_csv('/kaggle/input/fifa19/data.csv')
fifaDataset.head(5)
fifaDataset.describe() #this command show basic information about the attributes of the dataset

#For example, the mean age of the players in FIFA 19 is 25.12 years.
fifaDataset.shape #rows x columns
#First let's create a dataset with only japanese players

japanDataset=fifaDataset[(fifaDataset['Nationality'] == "Japan")]
#Let's check the possible values of Age attribute

sorted(list(set(fifaDataset['Age'].values)))
#The youngest age is 16. I'm going to select the players whose age goes from 16 to 21 years old.

youngJapanDataset=japanDataset[japanDataset.Age <=21]
#Let's check our modified dataset

youngJapanDataset.head()
#I'm going to make a dataset of players with a potential >=75

morePotentialJapanDataset=youngJapanDataset[youngJapanDataset.Potential >=75]
morePotentialJapanDataset.head()
fig = px.bar(morePotentialJapanDataset,

             x='Name',

             y='Potential',

             title='Young japanese players with most potential',

             color='Potential',

             barmode='stack')



fig.update_layout(

   paper_bgcolor='rgb(0, 0, 0)',

   plot_bgcolor='rgb(0, 0, 0)',

    font_family="Helvetica",

    font_color="white",

    title_font_family="Helvetica",

    title_font_color= "white",

    

    xaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    },

    yaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    }

    

)









# plot

fig.show()
#Let's check his stats in FIFA 19

statsTomiyasu=morePotentialJapanDataset[morePotentialJapanDataset.Name=="T. Tomiyasu"]



pd.set_option('display.max_columns', 500)#this way I can see the values for each column
statsTomiyasu 
statsIto=morePotentialJapanDataset[morePotentialJapanDataset.Name=="T. Itō"]

pd.set_option('display.max_columns', 500)#this way I can see the values for each column

statsIto
statsAbe=morePotentialJapanDataset[morePotentialJapanDataset.Name=="H. Abe"]

pd.set_option('display.max_columns', 500)#this way I can see the values for each column

statsAbe
statsKamada=morePotentialJapanDataset[morePotentialJapanDataset.Name=="D. Kamada"]

pd.set_option('display.max_columns', 500)#this way I can see the values for each column

statsKamada
fig = px.bar(morePotentialJapanDataset,

             x='Name',

             y='Overall',

             title='Young japanese players with most punctuation in the present',

             color='Overall',

             barmode='stack')



fig.update_layout(

   paper_bgcolor='rgb(0, 0, 0)',

   plot_bgcolor='rgb(0, 0, 0)',

    font_family="Helvetica",

    font_color="white",

    title_font_family="Helvetica",

    title_font_color= "white",

    

    xaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    },

    yaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    }

    

)









# plot

fig.show()
positions=youngJapanDataset.groupby('Position').size().reset_index(name='count')

fig = px.bar(positions,

             x='Position',

             y='count',

             title='Popular positions in the game',

             color='count',

             barmode='stack')



fig.update_layout(

   paper_bgcolor='rgb(0, 0, 0)',

   plot_bgcolor='rgb(0, 0, 0)',

    font_family="Helvetica",

    font_color="white",

    title_font_family="Helvetica",

    title_font_color= "white",

    

    xaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    },

    yaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    }

    

)









# plot

fig.show()

statsØdegaard=fifaDataset[(fifaDataset['Name'] == "M. Ødegaard" ) & (fifaDataset['Club'] == "Vitesse" ) ]# he played for the Vitesse at that moment

pd.set_option('display.max_columns', 500)

statsØdegaard
statsKubo=morePotentialJapanDataset[morePotentialJapanDataset.Name=="T. Kubo"]
comparation=pd.DataFrame()

comparation=comparation.append(statsØdegaard)

comparation=comparation.append(statsKubo)

comparation.dropna(axis=1, how='all')
fig = px.bar(comparation,

             x='Name',

             y='Potential',

             title='Potentials of Martin Ødegaard and Takefusa Kubo',

             color='Potential',

             barmode='stack')



fig.update_layout(

   paper_bgcolor='rgb(0, 0, 0)',

   plot_bgcolor='rgb(0, 0, 0)',

    font_family="Helvetica",

    font_color="white",

    title_font_family="Helvetica",

    title_font_color= "white",

    

    xaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    },

    yaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    }

    

)









# plot

fig.show()
statsNeres=fifaDataset[(fifaDataset['Name'] == "David Neres" )  & (fifaDataset['Club'] == "Ajax" )]
comparation2=pd.DataFrame()

comparation2=comparation2.append(statsIto)

comparation2=comparation2.append(statsNeres)

comparation2.dropna(axis=1, how='all')
#Comparation between goalkeepers

comparationGoalKeepers=pd.DataFrame()

statsCourtois=fifaDataset[(fifaDataset['Nationality'] == "Belgium"  ) & (fifaDataset['Name'] =="T. Courtois"  ) ]

statsKawashima=fifaDataset[(fifaDataset['Nationality'] == "Japan"  ) & (fifaDataset['Name'] =="E. Kawashima"  ) ]

comparationGoalKeepers=comparationGoalKeepers.append(statsCourtois)

comparationGoalKeepers=comparationGoalKeepers.append(statsKawashima)

comparationGoalKeepers.dropna(axis=1, how='all')
#Comparation between centre-backs:

#Comparation between Toby Alderweireld (in this game he played as right-back but his main position is centre-back ), Jan Vertoghen (in this game he played as left-back),

#Gen Shoji and Maya Yoshida.

comparationCentreBacks=pd.DataFrame()

statsVertonghen=fifaDataset[(fifaDataset['Nationality'] == "Belgium"  ) & (fifaDataset['Name'] =="J. Vertonghen"  ) ]

statsAlderweireld=fifaDataset[(fifaDataset['Nationality'] == "Belgium"  ) & (fifaDataset['Name'] =="T. Alderweireld"  ) ]



comparationCentreBacks=comparationCentreBacks.append(statsVertonghen)

comparationCentreBacks=comparationCentreBacks.append(statsAlderweireld)





statsShoji=fifaDataset[(fifaDataset['Nationality'] == "Japan"  ) & (fifaDataset['Name'] =="G. Shoji"  ) ]

statsYoshida=fifaDataset[(fifaDataset['Nationality'] == "Japan"  ) & (fifaDataset['Name'] =="M. Yoshida"  ) ]



comparationCentreBacks=comparationCentreBacks.append(statsShoji)

comparationCentreBacks=comparationCentreBacks.append(statsYoshida)



comparationCentreBacks

fig = px.bar(comparationCentreBacks,

             x='Name',

             y='Overall',

             title='Comparation between Belgian and Japanese players ',

             color='Overall',

             barmode='stack')



fig.update_layout(

   paper_bgcolor='rgb(0, 0, 0)',

   plot_bgcolor='rgb(0, 0, 0)',

    font_family="Helvetica",

    font_color="white",

    title_font_family="Helvetica",

    title_font_color= "white",

    

    xaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    },

    yaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    }

    

)









# plot

fig.show()
fig = px.bar(comparationCentreBacks,

             x='Name',

             y='Acceleration',

             title='Comparation between Belgian and Japanese players ',

             color='Acceleration',

             barmode='stack')



fig.update_layout(

   paper_bgcolor='rgb(0, 0, 0)',

   plot_bgcolor='rgb(0, 0, 0)',

    font_family="Helvetica",

    font_color="white",

    title_font_family="Helvetica",

    title_font_color= "white",

    

    xaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    },

    yaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    }

    

)









# plot

fig.show()
fig = px.bar(comparationCentreBacks,

             x='Name',

             y='Stamina',

             title='Comparation between Belgian and Japanese players ',

             color='Stamina',

             barmode='stack')



fig.update_layout(

   paper_bgcolor='rgb(0, 0, 0)',

   plot_bgcolor='rgb(0, 0, 0)',

    font_family="Helvetica",

    font_color="white",

    title_font_family="Helvetica",

    title_font_color= "white",

    

    xaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    },

    yaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    }

    

)









# plot

fig.show()
fig = px.bar(comparationCentreBacks,

             x='Name',

             y='Strength',

             title='Comparation between Belgian and Japanese players ',

             color='Strength',

             barmode='stack')



fig.update_layout(

   paper_bgcolor='rgb(0, 0, 0)',

   plot_bgcolor='rgb(0, 0, 0)',

    font_family="Helvetica",

    font_color="white",

    title_font_family="Helvetica",

    title_font_color= "white",

    

    xaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    },

    yaxis = { 

    'showgrid': False, 

    'zeroline': True, 

    'visible': True,

    

    }

    

)









# plot

fig.show()
from sklearn.model_selection import train_test_split 
#some parameters should be categorical

fifaDataset['Overall']=fifaDataset['Overall'].astype(str)

fifaDataset['Age']=fifaDataset['Age'].astype(str)
#separate target from predictors

y=fifaDataset.Overall

columns=['Age', 'Skill Moves', 'Work Rate', 'Position', 'Crossing', 'Finishing',

'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance','ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']

X=fifaDataset[columns]



#divide data into training and validation subsets

X_train_full, X_valid_full, y_train,y_valid=train_test_split(X,y,train_size=0.70, test_size=0.30,random_state=0)

categorical_columns=[column_name for column_name in X_train_full.columns if X_train_full[column_name].dtype=="object"]
numerical_columns=[column_name for column_name in X_train_full.columns if X_train_full[column_name].dtype in ["int64", "float64"]]
my_cols=categorical_columns+numerical_columns

X_train=X_train_full[my_cols].copy()

X_valid=X_valid_full[my_cols].copy()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='mean')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_columns),

        ('cat', categorical_transformer, categorical_columns)

    ])
from sklearn.ensemble import RandomForestClassifier

fifaModel = RandomForestClassifier(n_estimators=200, random_state=0)

pipelineFifa = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', fifaModel)

                             ])
pipelineFifa.fit(X_train, y_train)





predictions = pipelineFifa.predict(X_valid)
#evaluation of the model

from sklearn.metrics import mean_absolute_error

score = mean_absolute_error(y_valid, predictions)

print('MAE:', score)
#Predictions

print(predictions)