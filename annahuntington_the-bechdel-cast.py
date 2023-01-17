# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
bechdel_path = "/kaggle/input/bechdel3/BechdelCastRatingVersion3.csv"

bechdel_data = pd.read_csv(bechdel_path, engine = "python")

bechdel_data.columns
yes_count = 0



data = bechdel_data.bechdel_test



for i in data:

    if i == 'Yes':

        yes_count +=1

        

print(round(yes_count/len(data)*100, 2), "% of movies pass the Bechdel test (as of 9/23/2020)")
ave_rat = np.average(bechdel_data.average_rating)

#Might be a normal distribution we can explore that a little later



host_rat = np.average(bechdel_data.host_average)

print("The average rating is", ave_rat,"\nThe average host rating is", host_rat, "\nIs the difference in host ratings statistically significant?")



#ave_dev = np.std(bechdel_data.average_rating)

#host_dev = np.std(bechdel_data.host_average)

#print("The average rating standard deviation", ave_dev, "\nThe host average standard deviation", host_dev)
df = pd.DataFrame({'durante': bechdel_data.durante, 'loftus':bechdel_data.loftus, 'guest':bechdel_data.guest})

df['guest']= df['guest'].fillna(df['guest'].mean())

df['durante']= df['durante'].fillna(df['durante'].mean())



print(df)
%matplotlib inline



df.hist(bins=50)
ave_rat_median = np.median(bechdel_data.average_rating)



ave_rat_std = np.std(bechdel_data.average_rating)



print(ave_rat_median, ave_rat_std)
from scipy.stats import ttest_ind



ttest_ind(df.durante, df.loftus)
from scipy import stats



stats.ttest_rel(df.durante, df.loftus)
ttest_ind(df.durante, df.guest) 
ttest_ind(df.loftus, df.guest)
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



genre = pd.DataFrame({'genre': bechdel_data.main_genre, 'ave_rating':bechdel_data.average_rating, 'host_average':bechdel_data.host_average})



one_hot_training = pd.get_dummies(genre)



one_hot_training.head()



y = genre.ave_rating



genre_features = ['genre_Action', 'genre_Adventure', 'genre_Animated', 'genre_Comedy', 'genre_Drama', 'genre_Fantasy', 'genre_Horror', 'genre_Musical', 'genre_Rom-com', 'genre_Romance', 'genre_Sci-fi', 'genre_Thiller']



x = one_hot_training[genre_features]

x.describe()

x.head()



#split into x_val and y_val



rating_model = DecisionTreeRegressor()

rating_model.fit(x, y)



# Parasite (thriller), How Stella got her Groove back (Romance), Death Becomes Her (Comedy)

xnew = [[0,0,0,0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,0,1,0,0], [0,0,0,1,0,0,0,0,0,0,0,0]]



ynew = rating_model.predict(xnew)



print(ynew)
