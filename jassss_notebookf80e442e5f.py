{

 "cells": [

  {

   "cell_type": "code",

   "execution_count": 2,

   "metadata": {

    "collapsed": true

   },

   "outputs": [],

   "source": [

    "import json \n",

    "import pandas as pd\n",

    "import numpy as np\n",

    "from cPickle import dump,load"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": 31,

   "metadata": {

    "collapsed": false,

    "scrolled": true

   },

   "outputs": [

    {

     "data": {

      "text/plain": [

       "u'baking powder eggs all-purpose flour raisins milk white sugar'"

      ]

     },

     "execution_count": 31,

     "metadata": {},

     "output_type": "execute_result"

    }

   ],

   "source": [

    "test_data = open('../data/test/test.json').read()\n",

    "test = json.loads(test_data)\n",

    "Ids = [ d['id'] for d in test]\n",

    "Cuisines = [ 'italian' for d in test]\n",

    "Words = [' '.join(  d['ingredients'] \n",

    "               # +   [ word.strip().split()[-1] for word in d['ingredients'] if len(word.strip().split()) > 1 ]\n",

    "            ) for d in test]\n",

    "\n",

    "testdf= pd.DataFrame({\n",

    "    'Id' : Ids, \n",

    "    'Cuisine' : Cuisines,\n",

    "    'txt' : Words,\n",

    "\n",

    "})\n",

    "\n",

    "dump(testdf, open('testdf.pickle', 'wb'))\n",

    "\n",

    "testdf['txt'][0]\n",

    "\n"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": 10,

   "metadata": {

    "collapsed": false

   },

   "outputs": [

    {

     "data": {

      "text/plain": [

       "u'romaine lettuce black olives grape tomatoes garlic pepper purple onion seasoning garbanzo beans feta cheese crumbles'"

      ]

     },

     "execution_count": 10,

     "metadata": {},

     "output_type": "execute_result"

    }

   ],

   "source": [

    "json_data=open('../data/train/train.json').read()\n",

    "data = json.loads(json_data)\n",

    "\n",

    "Ids = [ d['id'] for d in data]\n",

    "Cuisines = [d['cuisine'] for d in data]\n",

    "\n",

    "# you can change how to parse the words in ingredients\n",

    "Words = [' '.join( d['ingredients']  \n",

    "               # + [ word.strip().split()[-1] for word in d['ingredients'] if len(word.strip().split()) > 1 ]\n",

    "                 ) for d in data]\n",

    "\n",

    "df= pd.DataFrame({\n",

    "    'Id' : Ids, \n",

    "    'Cuisine' : Cuisines,\n",

    "    'txt' : Words\n",

    "    \n",

    "})\n",

    "\n",

    "dump(df, open('df.pickle', 'wb'))\n",

    "df = load(open('df.pickle','rb'))\n",

    "df['txt'][0]"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": 14,

   "metadata": {

    "collapsed": false

   },

   "outputs": [

    {

     "data": {

      "text/plain": [

       "(39774, 3010)"

      ]

     },

     "execution_count": 14,

     "metadata": {},

     "output_type": "execute_result"

    }

   ],

   "source": [

    "from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer     \n",

    "vect = TfidfVectorizer(ngram_range=(1,1), max_features = 10000) #analyzer\n",

    "xvec = vect.fit_transform(df.txt)\n",

    "\n",

    "xvec.shape"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": 15,

   "metadata": {

    "collapsed": false

   },

   "outputs": [

    {

     "data": {

      "text/plain": [

       "(9944, 3010)"

      ]

     },

     "execution_count": 15,

     "metadata": {},

     "output_type": "execute_result"

    }

   ],

   "source": [

    "test_vect = vect.transform(testdf.txt)\n",

    "#test_vect2 = vect.transform(testdf.txt2)\n",

    "test_vect.shape\n"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": 16,

   "metadata": {

    "collapsed": false

   },

   "outputs": [

    {

     "name": "stdout",

     "output_type": "stream",

     "text": [

      "20\n"

     ]

    }

   ],

   "source": [

    "from sklearn.preprocessing import LabelEncoder\n",

    "le = LabelEncoder()\n",

    "label = le.fit_transform(df['Cuisine'])\n",

    "\n",

    "print len(set(label))"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": 17,

   "metadata": {

    "collapsed": false

   },

   "outputs": [

    {

     "data": {

      "text/plain": [

       "array([u'greek', u'southern_us', u'filipino', ..., u'irish', u'chinese',\n",

       "       u'mexican'], dtype=object)"

      ]

     },

     "execution_count": 17,

     "metadata": {},

     "output_type": "execute_result"

    }

   ],

   "source": [

    " le.inverse_transform(label)"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": 657,

   "metadata": {

    "collapsed": false

   },

   "outputs": [],

   "source": [

    "#df['Cuisine']"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": 18,

   "metadata": {

    "collapsed": false

   },

   "outputs": [

    {

     "name": "stdout",

     "output_type": "stream",

     "text": [

      "0.730998072572\n"

     ]

    }

   ],

   "source": [

    "from sklearn.cross_validation import train_test_split\n",

    "train_X, test_X, train_y, test_y = train_test_split(xvec, label , train_size=0.7, random_state=1)\n",

    "from sklearn.naive_bayes import MultinomialNB\n",

    "clf_nb = MultinomialNB(alpha=0.01, fit_prior = True)\n",

    "clf_nb.fit(train_X, train_y)\n",

    "from sklearn import metrics\n",

    "pre_nb = clf_nb.predict(test_X) \n",

    "print metrics.accuracy_score(test_y, pre_nb)"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": 20,

   "metadata": {

    "collapsed": false

   },

   "outputs": [

    {

     "name": "stdout",

     "output_type": "stream",

     "text": [

      "accuracy of linear SVM:  0.785468867845\n"

     ]

    }

   ],

   "source": [

    "from sklearn.svm import LinearSVC\n",

    "from sklearn.multiclass import OneVsRestClassifier\n",

    "clf_svm = OneVsRestClassifier(LinearSVC( C = 1, penalty = 'l2', dual = False))\n",

    "clf_svm.fit(train_X, train_y)\n",

    "from sklearn import metrics\n",

    "pre_svm = clf_svm.predict(test_X)\n",

    "print \"accuracy of linear SVM: \", metrics.accuracy_score(test_y, pre_svm)"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": 21,

   "metadata": {

    "collapsed": false

   },

   "outputs": [],

   "source": [

    "pred_cuisine = clf_svm.predict(test_vect)\n",

    "pred_cuisine\n",

    "pred_cuisine_label = le.inverse_transform(pred_cuisine)\n",

    "pred_cuisine_label\n",

    "submission = pd.read_csv('../sample_submission.csv')\n",

    "submission['cuisine'] = pred_cuisine_label\n",

    "submission['id'] = testdf['Id']\n",

    "\n",

    "submission.to_csv('linearsvm_benchmark.csv', index=False)"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": 22,

   "metadata": {

    "collapsed": false

   },

   "outputs": [

    {

     "name": "stdout",

     "output_type": "stream",

     "text": [

      "accuracy of Ridge classifier:  0.754546216375\n"

     ]

    }

   ],

   "source": [

    "from sklearn.linear_model import RidgeClassifier\n",

    "clf_rg = RidgeClassifier(tol=0.01, solver=\"lsqr\")\n",

    "clf_rg.fit(train_X, train_y)\n",

    "from sklearn import metrics\n",

    "pre_rg = clf_rg.predict(test_X)\n",

    "print \"accuracy of Ridge classifier: \", metrics.accuracy_score(test_y, pre_rg)"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": 23,

   "metadata": {

    "collapsed": false

   },

   "outputs": [],

   "source": [

    "#print \"accuracy of Lasso classifier: \", metrics.accuracy_score(test_y, pre_lo)"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": 25,

   "metadata": {

    "collapsed": false

   },

   "outputs": [

    {

     "name": "stdout",

     "output_type": "stream",

     "text": [

      "accuracy of Logistic Regression classifier:  0.783792843376\n"

     ]

    }

   ],

   "source": [

    "from sklearn.linear_model import LogisticRegression\n",

    "from sklearn.multiclass import OneVsRestClassifier\n",

    "clf_eln = OneVsRestClassifier(LogisticRegression(penalty='l2', dual=False, tol=0.0001 ,C = 3))\n",

    "clf_eln.fit(train_X, train_y)\n",

    "\n",

    "pre_eln = clf_eln.predict(test_X)\n",

    "print \"accuracy of Logistic Regression classifier: \", metrics.accuracy_score(test_y, pre_eln)"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": 26,

   "metadata": {

    "collapsed": false

   },

   "outputs": [

    {

     "name": "stdout",

     "output_type": "stream",

     "text": [

      "0.785133662951\n"

     ]

    }

   ],

   "source": [

    "from sklearn.linear_model import SGDClassifier\n",

    "clf_sgd = SGDClassifier(loss='log', penalty='elasticnet', alpha=0.00001, l1_ratio=0.55, n_iter=100, shuffle=True, \n",

    "                        epsilon=1e-2, n_jobs=4, learning_rate='optimal', eta0=0.0)\n",

    "clf_sgd.fit(train_X, train_y)\n",

    "\n",

    "pre_sgd = clf_sgd.predict(test_X)\n",

    "print metrics.accuracy_score(test_y, pre_sgd)"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": 27,

   "metadata": {

    "collapsed": true

   },

   "outputs": [],

   "source": [

    "pred_cuisine = clf_sgd.predict(test_vect)\n",

    "pred_cuisine\n",

    "pred_cuisine_label = le.inverse_transform(pred_cuisine)\n",

    "pred_cuisine_label\n",

    "submission = pd.read_csv('../sample_submission.csv')\n",

    "submission['cuisine'] = pred_cuisine_label\n",

    "submission['id'] = testdf['Id']\n",

    "submission.to_csv('sgd_benchmark.csv', index=False)"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": 30,

   "metadata": {

    "collapsed": false

   },

   "outputs": [

    {

     "name": "stdout",

     "output_type": "stream",

     "text": [

      "accuracy of random forest with 100 trees is :  0.739713399816\n"

     ]

    }

   ],

   "source": [

    "from sklearn.grid_search import GridSearchCV, RandomizedSearchCV\n",

    "from sklearn.cross_validation import cross_val_score\n",

    "from sklearn import ensemble\n",

    "from time import time\n",

    "from sklearn.ensemble import RandomForestClassifier\n",

    "# Random forest\n",

    "rf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators = 100)\n",

    "\n",

    "rf.fit(train_X.toarray(), train_y)\n",

    "pre = rf.predict(test_X.toarray())\n",

    "print \"accuracy of random forest with 100 trees is : \", metrics.accuracy_score(test_y, pre)"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": null,

   "metadata": {

    "collapsed": false,

    "scrolled": true

   },

   "outputs": [],

   "source": [

    "import xgboost as xgb\n",

    "xg_train = xgb.DMatrix(train_X, label=train_y)\n",

    "xg_test = xgb.DMatrix(test_X, label=test_y)\n",

    "\n",

    "# setup parameters for xgboost"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": null,

   "metadata": {

    "collapsed": false,

    "scrolled": false

   },

   "outputs": [],

   "source": [

    "param = {}\n",

    "#use softmax multi-class classification\n",

    "param['objective'] = 'multi:softmax'\n",

    "# scale weight of positive examples\n",

    "param['eta'] = 0.01\n",

    "param['max_depth'] = 7\n",

    "param['silent'] = 1\n",

    "#param['min_child_weight'] = 5\n",

    "param[\"subsample\"] = 0.7\n",

    "param[\"colsample_bytree\"] = 0.7\n",

    "param['nthread'] = 4\n",

    "param['num_class'] = 20\n",

    "\n",

    "watchlist = [ (xg_train,'train'), (xg_test, 'test') ]\n",

    "num_round = 10\n",

    "bst = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=80);\n",

    "# get prediction\n",

    "pred = bst.predict(xg_test);\n",

    "print ('predicting, classification error=%f' % (sum( int(pred[i]) != test_y[i] for i in range(len(test_y))) / float(len(test_y)) ))\n"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": 58,

   "metadata": {

    "collapsed": false

   },

   "outputs": [

    {

     "data": {

      "text/plain": [

       "array([  1.,  16.,   9., ...,   9.,  16.,  13.], dtype=float32)"

      ]

     },

     "execution_count": 58,

     "metadata": {},

     "output_type": "execute_result"

    }

   ],

   "source": [

    "#xg_test = xgb.DMatrix(test_X, label=test_y)\n",

    "pred_cuisine = bst.predict(xgb.DMatrix(test_vect))\n",

    "pred_cuisine\n"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": 59,

   "metadata": {

    "collapsed": false

   },

   "outputs": [],

   "source": [

    "pred_cuisine_label = le.inverse_transform(np.array(pred_cuisine, dtype = int))\n",

    "pred_cuisine_label\n",

    "submission = pd.read_csv('../sample_submission.csv')\n",

    "submission['cuisine'] = pred_cuisine_label\n",

    "submission['id'] = testdf['Id']\n",

    "submission.to_csv('xgb_benchmark.csv', index=False)"

   ]

  },

  {

   "cell_type": "code",

   "execution_count": null,

   "metadata": {

    "collapsed": true

   },

   "outputs": [],

   "source": []

  }

 ],

 "metadata": {

  "kernelspec": {

   "display_name": "Python 2",

   "language": "python",

   "name": "python2"

  },

  "language_info": {

   "codemirror_mode": {

    "name": "ipython",

    "version": 2

   },

   "file_extension": ".py",

   "mimetype": "text/x-python",

   "name": "python",

   "nbconvert_exporter": "python",

   "pygments_lexer": "ipython2",

   "version": "2.7.9"

  }

 },

 "nbformat": 4,

 "nbformat_minor": 0

}
