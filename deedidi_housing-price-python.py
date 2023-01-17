# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_data=pd.read_csv("../input/train.csv")

test_data=pd.read_csv("../input/test.csv")
import numpy as np

import pandas as pd
train_Numeric = train_data.select_dtypes(include=[np.number]) 

train_Category=train_data.select_dtypes(exclude=[np.number]) 

cat_dict = train_Category.to_dict(orient = 'records')



from sklearn.feature_extraction import DictVectorizer as DV

vectorizer = DV( sparse = False )

vec_x_cat_train = vectorizer.fit_transform( cat_dict )

#vectorizer.get_feature_names()



np.nan_to_num(vec_x_cat_train)
train_data = pd.get_dummies(train_data)

train_data = train_data.fillna(0)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(train_data, train_data['SalePrice'])

                  