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
!pip install ktrain
d1 = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")

d1["fake"] = False



d = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")

d["fake"] = True



df = pd.concat([d, d1])

df.head()
X = np.asarray(df["text"])

y = np.asarray(df["fake"])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

import ktrain

from ktrain import text

MODEL_NAME = 'distilbert-base-uncased'

t = text.Transformer(MODEL_NAME, maxlen=500, class_names=["True", "False"])

trn = t.preprocess_train(X_train, y_train)

val = t.preprocess_test(X_test, y_test)

model = t.get_classifier()

learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=6)
learner.fit_onecycle(8e-5, 1)
learner.validate(class_names=["True", "False"])
learner.view_top_losses(n=1, preproc=t)
predictor = ktrain.get_predictor(learner.model, preproc=t)

predictor.predict(X_test[7399])

predictor.explain(X_test[7399])
predictor.explain(X_test[10])
predictor.explain(X_test[100])