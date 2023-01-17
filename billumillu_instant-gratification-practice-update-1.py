# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.





from sklearn.metrics import mean_absolute_error,accuracy_score

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier,XGBRegressor
X_full = pd.read_csv("../input/train.csv")

X_test_full = pd.read_csv("../input/test.csv")

print("Loaded.")
#X_full.shape  

#(262144, 258)

#X_test.shape

#(131073, 257)



X_full.head(10)
#X_full.isnull().sum() #it's not showing some columns... how do  know for sure?

cols_missingvals = [col for col in X_full.columns if X_full[col].isnull().any()]

print(cols_missingvals)


y = X_full.target

X = X_full.drop(['target','id'], axis=1)

X_test = X_test_full.drop(['id'], axis=1)



X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)



my_model = XGBRegressor(n_estimators=100, learning_rate=0.05)

my_model.fit(X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(X_valid, y_valid)], 

             verbose=False)

predictions = my_model.predict(X_valid)

print("MAE: " + str(mean_absolute_error(predictions, y_valid)))

# preds = my_model.predict(X_valid)

# preds_test = my_model.predict(X_test)

# preds_test_rounded = np.around(preds_test,decimals=1)



# output = pd.DataFrame({'id': X_test_full.id,

#                        'target': preds_test_rounded})

# output.to_csv('submission.csv', index=False)



# a = pd.read_csv('submission.csv')

# a.head(10)

# b = pd.read_csv('../input/sample_submission.csv')

# b.head(10)
import matplotlib.pyplot as plt



#plot bar chart with matplotlib

plt.figure(figsize=(17,10))



y_pos = np.arange(len(X.columns))



plt.bar(y_pos, my_model.feature_importances_, align='center', alpha=0.5)

plt.xticks(y_pos, X.columns)

plt.xticks(rotation=90)



plt.xlabel('Features')

plt.ylabel('Feature Importance')



plt.title('Feature importances')



plt.show()
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



data = [go.Bar(

            x=y_pos,

            y=my_model.feature_importances_

    )]



#iplot(data)



layout = go.Layout(

    xaxis = go.layout.XAxis(

        tickmode = 'array',

        tickvals = y_pos,

        ticktext = X.columns,

        tickangle = -90

    )

)



fig = go.Figure(

    data = data,

    layout = layout

)



iplot(fig)
features = ['bluesy-rose-wallaby-discard','cranky-cardinal-dogfish-ordinal','homey-sepia-bombay-sorted','hasty-blue-sheep-contributor',

            'blurry-wisteria-oyster-master','baggy-mustard-collie-hint','beady-champagne-bullfrog-grandmaster','blurry-flax-sloth-fepid',

           'grumpy-zucchini-kudu-kernel','bluesy-amber-walrus-fepid','hazy-tan-schnauzer-hint','gloppy-turquoise-quoll-goose',

            'snoopy-red-zonkey-unsorted','snappy-brass-malamute-entropy','squeaky-khaki-lionfish-distraction',

            'crappy-pumpkin-saola-grandmaster','wheezy-harlequin-earwig-gaussian','tasty-buff-monkey-learn','dorky-turquoise-maltese-important',

           'hasty-puce-fowl-fepid','stuffy-periwinkle-zebu-discard','breezy-myrtle-loon-discard','woolly-gold-millipede-fimbus',

           'bluesy-amethyst-octopus-gaussian','dorky-cream-flamingo-novice','gimpy-asparagus-eagle-novice','stealthy-yellow-lobster-goose',

           'freaky-olive-insect-ordinal','greasy-scarlet-paradise-goose','pretty-copper-insect-discard','gloppy-buff-frigatebird-dataset',

           'wheezy-lavender-catfish-master','cheeky-pear-horse-fimbus','stinky-olive-kiwi-golden','stealthy-azure-gopher-hint',

            'sleazy-russet-iguana-unsorted','surly-corn-tzu-kernel','woozy-apricot-moose-hint','greasy-magnolia-spider-grandmaster',

           'chewy-bistre-buzzard-expert','wheezy-myrtle-mandrill-entropy','muggy-turquoise-donkey-important','blurry-buff-hyena-entropy']
y = X_full.target

X = X_full[features]

X_test = X_test_full[features]

X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)
# my_model = XGBClassifier(n_estimators=100, learning_rate=0.05)

# my_model.fit(X_train, y_train, 

#              early_stopping_rounds=5, 

#              eval_set=[(X_valid, y_valid)], 

#              verbose=False)

# predictions = my_model.predict(X_valid)

# print("Accuracy Score: " + str(accuracy_score(predictions, y_valid)))



#Accuracy Score: 0.5095271700776288
my_model = XGBRegressor(n_estimators=100, learning_rate=0.05)

my_model.fit(X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(X_valid, y_valid)], 

             verbose=False)

predictions = my_model.predict(X_valid)

print("MAE: " + str(mean_absolute_error(predictions, y_valid)))
preds = my_model.predict(X_valid)

preds_test = my_model.predict(X_test)

preds_test_rounded = np.around(preds_test,decimals=1)



output = pd.DataFrame({'id': X_test_full.id,

                       'target': preds_test_rounded})

output.to_csv('submission.csv', index=False)
