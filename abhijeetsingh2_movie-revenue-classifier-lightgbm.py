import pandas as pd

data = pd.read_csv("data/train.csv")

data.head()
data.describe()
# Preprocessing



def clean_ratings_metacritic (x):

    x=int(x.split('/')[0])

    if x==0:

        x = None

    return x



def cleanse(data, isSubmission=False):  

    data.users_votes=data.users_votes.apply(lambda x: float(x.replace(',','')))

    data.ratings_imdb=data.ratings_imdb.apply(lambda x: float(x.split('/')[0])*10)

#     data.ratings_metacritic=data.ratings_metacritic.apply(clean_ratings_metacritic)

    data.ratings_tomatoes=data.ratings_tomatoes.apply(lambda x: float(x.split('%')[0]))



cleanse(data)

data.describe()
data.dropna(inplace=True)

data.describe()
corr=data.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
from sklearn.model_selection import train_test_split



drop_val=['title','country','genres','language','writer_count','title_adaption','censor_rating','release_date','runtime','dvd_release_date', 'special_award','ratings_metacritic', 'revenue_category']

train=data.drop(drop_val, inplace=False, axis=1)

y = data.revenue_category.values

x=train.values

x, x_test, y, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
from sklearn import metrics

import lightgbm



# fit a lightGBM classifier to the data

model = lightgbm.LGBMClassifier(objective='binary', learning_rate=0.02, n_estimators=200)  #using binary classifier as we have 2 output classes

model.fit(x, y,

          eval_set=[(x_test, y_test)],          

          early_stopping_rounds=5)



print (); print(model)
expected_y  = y_test

predicted_y = model.predict(x_test)



print(); print(metrics.classification_report(expected_y, predicted_y))
import matplotlib.pyplot as plt



lightgbm.plot_importance(model, title='Important Features ranked on the basis of FIS (Feature Importance Score)')

plt.show()

train.head()
submission = pd.read_csv('data/test.csv')

title = submission['title'].values

drop_val.remove('revenue_category')

submission.drop(drop_val, inplace=True, axis=1)  #drop columns not to be used for prediction

cleanse(submission)  #pre-process submission data for prediction



x = submission.values

y = model.predict(x)



output = pd.DataFrame({'title': title, 'revenue_category': y})

output.to_csv("submissions.csv", index=False)
output.head()