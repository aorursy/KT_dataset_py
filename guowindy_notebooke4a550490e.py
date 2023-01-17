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
import pandas as pd

from sklearn import ensemble



if __name__ == "__main__":

    loc_train = "../input/train.csv"

    loc_test = "../input/test.csv"

    loc_submission = "kaggle.rf200.entropy.submission.csv"

  

    df_train = pd.read_csv(loc_train)

    df_test = pd.read_csv(loc_test)

  

    feature_cols = [col for col in df_train.columns if col not in ['Cover_Type','Id']]

  

    X_train = df_train[feature_cols]

    X_test = df_test[feature_cols]

    y = df_train['Cover_Type']

    test_ids = df_test['Id']

    del df_train

    del df_test

    

    clf = ensemble.RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=0)

    clf.fit(X_train, y)

    del X_train

    

    with open(loc_submission, "w") as outfile:

      outfile.write("Id,Cover_Type\n")

      for e, val in enumerate(list(clf.predict(X_test))):

        outfile.write("%s,%s\n"%(test_ids[e],val))