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
{

  "cells": [

    {

      "cell_type": "markdown",

      "metadata": {

        "_cell_guid": "39f3abfd-a593-224a-c1a0-5e8414444e82"

      },

      "source": [

        "1. Data/Libraries Load \n",

        "2. Data Preparation (Binarizing Categorical Attributes/ Train and Test Data Split Preperation)\n",

        "3. Model Fitting on Trained Data \n",

        "4. Parameters Tuning \n",

        "5. Predictions on Unseen Data \n",

        "6. Predictions on test.csv"

      ]

    },

    {

      "cell_type": "code",

      "execution_count": null,

      "metadata": {

        "_cell_guid": "610d391d-ef54-4018-e7d5-75e7dddb9e55"

      },

      "outputs": [],

      "source": [

        "from pandas import read_csv\n",

        "from xgboost import XGBRegressor\n",

        "from sklearn.cross_validation import train_test_split\n",

        "from sklearn.metrics import accuracy_score, mean_absolute_error\n",

        "import numpy \n",

        "from sklearn.grid_search import GridSearchCV\n",

        "# load data\n",

        "data = read_csv('../input/train.csv')\n",

        "data.head()"

      ]

    },

    {

      "cell_type": "code",

      "execution_count": null,

      "metadata": {

        "_cell_guid": "4f68feee-19fc-fb3b-72d7-c369c711a8fc"

      },

      "outputs": [],

      "source": [

        "# Binarizing Categorical Variables\n",

        "import pandas as pd\n",

        "features = data.columns\n",

        "cats = [feat for feat in features if 'cat' in feat]\n",

        "for feat in cats:\n",

        "    data[feat] = pd.factorize(data[feat], sort=True)[0]\n",

        "    \n",

        "data.head()"

      ]

    },

    {

      "cell_type": "code",

      "execution_count": null,

      "metadata": {

        "_cell_guid": "17474536-c074-d404-ba84-3b96f02a6435"

      },

      "outputs": [],

      "source": [

        "# Preparing data for train and test split \n",

        "x=data.drop(['id','loss'],1).fillna(value=0)\n",

        "y=data['loss']"

      ]

    },

    {

      "cell_type": "code",

      "execution_count": null,

      "metadata": {

        "_cell_guid": "0c5b8c6b-5411-7a9c-dbab-1034b763cc9c"

      },

      "outputs": [],

      "source": [

        "#Train and Test Split \n",

        "X_train, X_test, y_train, y_test = train_test_split( x.values, y.values, test_size=0.33, random_state=7)"

      ]

    },

    {

      "cell_type": "code",

      "execution_count": null,

      "metadata": {

        "_cell_guid": "61d6bc98-03d8-14d4-5967-63ac56501b1d"

      },

      "outputs": [],

      "source": [

        "# fitting model on training data\n",

        "model = XGBRegressor(max_depth=6, n_estimators=500, learning_rate=0.1, subsample=0.8, colsample_bytree=0.4,\n",

        "                     min_child_weight = 3,  seed=7)\n",

        "model.fit(X_train, y_train)\n",

        "print(model)\n",

        "#Making predictions\n",

        "y_pred = model.predict(X_test) \n",

        "predictions = [round(value) for value in y_pred]\n",

        "# evaluat predictions \n",

        "actuals = y_test\n",

        "print(mean_absolute_error(actuals, predictions))\n",

        "print(model.score(X_test, y_test)) "

      ]

    },

    {

      "cell_type": "code",

      "execution_count": null,

      "metadata": {

        "_cell_guid": "dcbbad3e-a92b-1f40-0ac6-e6b17020dc99"

      },

      "outputs": [],

      "source": [

        "Data_Test = read_csv('../input/test.csv')\n",

        "print(Data_Test.head())\n",

        "kfeatures = Data_Test.columns\n",

        "cats = [feat for feat in features if 'cat' in feat]\n",

        "for feat in cats:\n",

        "    Data_Test[feat] = pd.factorize(Data_Test[feat], sort=True)[0]\n",

        "    \n",

        "Test_X1 = Data_Test.drop(['id'],1).fillna(value=0)\n",

        "Test_X = Test_X1.values\n",

        "Data_Test['loss'] = model.predict(Test_X)\n",

        "\n",

        "Final_Result = Data_Test[['id','loss']]\n",

        "print(Final_Result.info())\n",

        "print(Final_Result.head())"

      ]

    },

    {

      "cell_type": "code",

      "execution_count": null,

      "metadata": {

        "_cell_guid": "45ec1437-535c-752b-4fc6-3e0a7b6284d8"

      },

      "outputs": [],

      "source": [

        "#Saving results to csv file\n",

        "Final_Result.to_csv('result.csv', index=False)"

      ]

    }

  ],

  "metadata": {

    "_change_revision": 0,

    "_is_fork": false,

    "kernelspec": {

      "display_name": "Python 3",

      "language": "python",

      "name": "python3"

    },

    "language_info": {

      "codemirror_mode": {

        "name": "ipython",

        "version": 3

      },

      "file_extension": ".py",

      "mimetype": "text/x-python",

      "name": "python",

      "nbconvert_exporter": "python",

      "pygments_lexer": "ipython3",

      "version": "3.5.2"

    }

  },

  "nbformat": 4,

  "nbformat_minor": 0

}