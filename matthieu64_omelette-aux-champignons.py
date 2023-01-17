import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns



%matplotlib inline
data= pd.read_csv("../input/mushroom-classification/mushrooms.csv")
data.shape
data.dtypes
data.head()
data.columns
#Number of unique values

########################################

#tab=[]

#for i in range(0,8124):

#    tab.append(i)

    





#data=data.assign(Random=tab)

########################################



cols=data.columns



for col in cols:

    print(col,' : ',data[col].unique())
# Get dummies

data = pd.get_dummies(data, prefix_sep='_', drop_first=True)

# X head

data.head()
data.columns
sns.set(rc={'figure.figsize':(30,20)})



correlation_matrix= np.corrcoef(data[data.columns].values.T)



hm= sns.heatmap(data=correlation_matrix, square=True, annot=True, fmt='.2f')
from sklearn.model_selection import train_test_split

#features pour évaluer

X=data[['cap-shape_c', 'cap-shape_f', 'cap-shape_k', 'cap-shape_s',

       'cap-shape_x', 'cap-surface_g', 'cap-surface_s', 'cap-surface_y',

       'cap-color_c', 'cap-color_e', 'cap-color_g', 'cap-color_n',

       'cap-color_p', 'cap-color_r', 'cap-color_u', 'cap-color_w',

       'cap-color_y', 'bruises_t', 'odor_c', 'odor_f', 'odor_l', 'odor_m',

       'odor_n', 'odor_p', 'odor_s', 'odor_y', 'gill-attachment_f',

       'gill-spacing_w', 'gill-size_n', 'gill-color_e', 'gill-color_g',

       'gill-color_h', 'gill-color_k', 'gill-color_n', 'gill-color_o',

       'gill-color_p', 'gill-color_r', 'gill-color_u', 'gill-color_w',

       'gill-color_y', 'stalk-shape_t', 'stalk-root_b', 'stalk-root_c',

       'stalk-root_e', 'stalk-root_r', 'stalk-surface-above-ring_k',

       'stalk-surface-above-ring_s', 'stalk-surface-above-ring_y',

       'stalk-surface-below-ring_k', 'stalk-surface-below-ring_s',

       'stalk-surface-below-ring_y', 'stalk-color-above-ring_c',

       'stalk-color-above-ring_e', 'stalk-color-above-ring_g',

       'stalk-color-above-ring_n', 'stalk-color-above-ring_o',

       'stalk-color-above-ring_p', 'stalk-color-above-ring_w',

       'stalk-color-above-ring_y', 'stalk-color-below-ring_c',

       'stalk-color-below-ring_e', 'stalk-color-below-ring_g',

       'stalk-color-below-ring_n', 'stalk-color-below-ring_o',

       'stalk-color-below-ring_p', 'stalk-color-below-ring_w',

       'stalk-color-below-ring_y', 'veil-color_o', 'veil-color_w',

       'veil-color_y', 'ring-number_o', 'ring-number_t', 'ring-type_f',

       'ring-type_l', 'ring-type_n', 'ring-type_p', 'spore-print-color_h',

       'spore-print-color_k', 'spore-print-color_n', 'spore-print-color_o',

       'spore-print-color_r', 'spore-print-color_u', 'spore-print-color_w',

       'spore-print-color_y', 'population_c', 'population_n', 'population_s',

       'population_v', 'population_y', 'habitat_g', 'habitat_l', 'habitat_m',

       'habitat_p', 'habitat_u', 'habitat_w']]

#Attribut à prédire

y=data['class_p']



 



xtrain, xtest, ytrain, ytest = train_test_split(X,y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression



logit=LogisticRegression()

logit.fit(xtrain,ytrain)



#ERROR

error = (1 - logit.score(xtest, ytest))*100

print('Erreur: %f' % error, '%')
print('Affichons les champignons où l\'on a fait une erreur.')

y_pred = logit.predict(xtest)

df = pd.DataFrame({'Actual': ytest, 'Predicted': y_pred})  

df.loc[df['Predicted'] != df['Actual']]