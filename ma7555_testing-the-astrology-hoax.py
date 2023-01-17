# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC



from sklearn.model_selection import train_test_split

athletes = pd.read_csv('../input/athletes.csv')

athletes.head()
def zodiac_transform(birthday):

    if birthday.month == 1:

        astro_sign = 'capricorn' if (birthday.day < 20) else 'aquarius'

    elif birthday.month == 2:

        astro_sign = 'aquarius' if (birthday.day < 19) else 'pisces'

    elif birthday.month == 3:

        astro_sign = 'pisces' if (birthday.day < 21) else 'aries'

    elif birthday.month == 4:

        astro_sign = 'aries' if (birthday.day < 20) else 'taurus'

    elif birthday.month == 5:

        astro_sign = 'taurus' if (birthday.day < 21) else 'gemini'

    elif birthday.month == 6:

        astro_sign = 'gemini' if (birthday.day < 21) else 'cancer'

    elif birthday.month == 7:

        astro_sign = 'cancer' if (birthday.day < 23) else 'leo'

    elif birthday.month == 8:

        astro_sign = 'leo' if (birthday.day < 23) else 'virgo'

    elif birthday.month == 9:

        astro_sign = 'virgo' if (birthday.day < 23) else 'libra'

    elif birthday.month == 10:

        astro_sign = 'libra' if (birthday.day < 23) else 'scorpio'

    elif birthday.month == 11:

        astro_sign = 'scorpio' if (birthday.day < 22) else 'sagittarius'

    elif birthday.month == 12:

        astro_sign = 'sagittarius' if (birthday.day < 22) else 'capricorn'

    return astro_sign



athletes.dropna(subset=['dob'], inplace=True)

athletes['dob'] = pd.to_datetime(athletes.dob)

athletes['zodiac'] = athletes.dob.apply(zodiac_transform)
athletes['zodiac_code'] = LabelEncoder().fit_transform(athletes.zodiac)

athletes['sex_code'] = LabelEncoder().fit_transform(athletes.sex)

athletes = pd.get_dummies(athletes, prefix_sep='_', columns=['sport'])

athletes.head()
features = athletes.drop(['id', 'name', 'nationality', 'sex', 'dob', 'height', 'weight', 'zodiac', 'zodiac_code'], axis=1)

labels = athletes.zodiac_code
# Splitting into train sets and test sets

x_train, x_test, y_train, y_test = train_test_split(features, labels, 

                                                    train_size=0.8,

                                                    test_size=0.2)
classifier = KNeighborsClassifier(n_neighbors=200).fit(x_train, y_train)

classifier.score(x_test, y_test)

# Nothing interesting here...
classifier = DecisionTreeClassifier(criterion='entropy', max_depth=4).fit(x_train, y_train)

classifier.score(x_test, y_test)

# Nothing here too...
classifier = SVC(kernel = 'linear', C=1).fit(x_train, y_train)

classifier.score(x_test, y_test)

# And nothing here as well...