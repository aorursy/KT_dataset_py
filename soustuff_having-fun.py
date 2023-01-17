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
chars = pd.read_csv("../input/simpsons_characters.csv")
episodes = pd.read_csv("../input/simpsons_episodes.csv")

#weird fix, check later
script = pd.read_csv("../input/simpsons_script_lines.csv",
                    error_bad_lines=False,
                    warn_bad_lines=False,
                    low_memory=False)


chars = chars.sort_values("id")
script = script.sort_values("episode_id")
script = script[["id","episode_id","character_id","timestamp_in_ms"]]
script = script.dropna()

episodes = episodes.sort_values("id")
episodes = episodes[["id","imdb_rating"]]
episodes.imdb_rating.unique()
columns = []
columns+=  [x for x in chars.id.values]
columns.append("score")
data = pd.DataFrame(columns = columns, index = list(script.episode_id.unique()))
data = data.fillna(0)
count = 0
for index, row in script.iterrows():
    #print(int(row.episode_id), int(row.character_id),int(row.timestamp_in_ms))
    try: #weird hack for poorly parse data
        data.loc[int(row.episode_id), int(row.character_id)] += int(row.timestamp_in_ms)
    except ValueError:
        try:
            print("error at index: ", index, " | Data: ", row.episode_id, row.character_id)
        except:
            print("unkown at index: ", index, " | Data: ", row.episode_id, row.character_id)
    else:
        pass #I dunno
        
print("done!") #for my mental sake
data.head()
for index, row in episodes.iterrows():
    try: #weird hack for poorly parse data
        discrete_score = np.NaN
        if row.imdb_rating >= 9:
            discrete_score = "very-good"
        elif row.imdb_rating < 9 and row.imdb_rating >= 7:
            discrete_score = "good"
        elif row.imdb_rating < 7 and row.imdb_rating >= 5:
            discrete_score = "regular"
        elif row.imdb_rating < 5 and row.imdb_rating >= 3:
            discrete_score = "bad"
        elif row.imdb_rating < 5 and row.imdb_rating >= 3:
            discrete_score = "very-bad"
            
        data.loc[int(row.id), "score"] = discrete_score
        
    except ValueError:
        print("error at index: ", index, " | Data: ", row.imdb_rating)


data = data.dropna()
from sklearn import svm
from sklearn.model_selection import cross_val_score
classifier = svm.SVC(gamma=0.01, C=100.)
target = data["score"]
train = data.drop(["score"], axis=1)
print(target.shape, train.shape)
scores = cross_val_score(classifier, train, target, cv = 40)
scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
