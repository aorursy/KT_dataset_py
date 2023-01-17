# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import re
import seaborn as sns
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
%matplotlib inline 

sns.set(style="white", color_codes=True)

pd.set_option('display.precision', 5)

df = pd.read_csv('../input/data.csv', encoding='utf-8-sig')
training, test = train_test_split(df, test_size = 0.2)


# Plot Hits and misses for each co-ordinate
court_scale, alpha = 7, 0.05
plt.figure(figsize=(2 * court_scale, court_scale*(84.0/50.0)))

# hit
plt.subplot(121)
hit = df.loc[df.shot_made_flag == 1]
plt.scatter(hit.loc_x, hit.loc_y, color='green', alpha=alpha)
plt.title('Hits')
ax = plt.gca()
ax.set_ylim([-50, 900])

# miss
plt.subplot(122)
miss = df.loc[df.shot_made_flag == 0]
plt.scatter(miss.loc_x, miss.loc_y, color='red', alpha=alpha)
plt.title('Misses')
ax = plt.gca()
ax.set_ylim([-50, 900])



# Scatter 
sns.set_style('white')
sns.set_color_codes()
plt.figure(figsize=(12,11))
plt.scatter(df['loc_x'],df['loc_y'])
plt.xlim(300,-300)
plt.ylim(-100,500)

#Court's outlines
from matplotlib.patches import Circle, Rectangle, Arc

def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()


    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box of the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the 
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250 , -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax

# let's draw the court
plt.figure(figsize=(12,11))
plt.scatter(df['loc_x'],df['loc_y'])
draw_court(outer_lines=True)

# and now draw the shots# Shooting accuracy with shot distance
def get_acc(df, against):
    ct = pd.crosstab(df.shot_made_flag, df[against]).apply(lambda x:x/x.sum(), axis=0)
    x, y = ct.columns, ct.values[1, :]
    plt.figure(figsize=(7, 5))
    plt.plot(x, y)
    plt.xlabel(against)
    plt.ylabel('% shots made')
    plt.savefig(against + '_vs_accuracy.png')
plt.ylim(-100,500)
plt.xlim(300,-300)

plt.show()

#Feature selection using #RandomForestClassifier

features_data = df[['loc_x', 'loc_y', 'minutes_remaining', 'lon','shot_distance','shot_made_flag']]
features_data = features_data.dropna()
X=features_data.drop('shot_made_flag', 1)
Y=features_data.shot_made_flag
names=features_data.dtypes.index

clf = RandomForestClassifier(n_jobs=-1)  

scores = []
for i in range(X.shape[1]):
     score = cross_val_score(clf, X, Y, scoring="roc_auc", cv=10)
     scores.append((round(np.mean(score), 3), names[i]))
print (sorted(scores, reverse=True))

#Training the model using Top 2 features 

clf = RandomForestClassifier(n_jobs=-1, n_estimators=500) 
train = df.loc[~df.shot_made_flag.isnull(), ['minutes_remaining',
                                             'shot_distance', 'shot_made_flag']]
test = df.loc[df.shot_made_flag.isnull(), ['minutes_remaining',
                                           'shot_distance', 'shot_id']]

# Train and predict
clf.fit(train.drop('shot_made_flag', 1), train.shot_made_flag)
predictions = clf.predict_proba(test.drop('shot_id', 1))

submission = pd.DataFrame({'shot_id': test.shot_id,
                           'shot_made_flag': predictions[:, 1]})
submission[['shot_id', 'shot_made_flag']].to_csv('submission.csv', index=False)

