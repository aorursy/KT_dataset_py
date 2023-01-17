import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from IPython.display import display, HTML
# Load dataFrame



train_df = pd.read_csv('../input/train.csv');

test_df = pd.read_csv('../input/test.csv');

display(train_df.head())

men_survived = train_df['Survived'][train_df['Sex'] == 'male'].value_counts()

women_survived = train_df['Survived'][train_df['Sex'] == 'female'].value_counts()



print(men_survived)

print(women_survived)



fig1, (alt1, alt2) = plt.subplots(1, 2)

labels = ["DEAD", "ALIVE"]



alt1.pie(men_survived,startangle=90,autopct='%1.1f%%' ,labels=[labels[i] for i in men_survived.keys()],explode=(0, 0.1))

alt1.axis('equal')

alt1.set_title("Men")



alt2.pie(women_survived, startangle=90, autopct="%1.1f%%", labels=[labels[i] for i in women_survived.keys()], explode=(0, 0.1), shadow=True)

alt2.axis('equal')

alt2.set_title("Women")



plt.show()
child_survived = train_df['Survived'][train_df['Age'] < 15].value_counts()

fig, alt = plt.subplots()

display(child_survived)

alt.pie(child_survived, labels = [labels[i] for i in child_survived.keys()], shadow=True, explode=(0,0.1), autopct="%1.1f%%")

alt.axis("equal")

plt.show()
cabin_wise = train_df['Cabin'][train_df['Survived'] == 1].value_counts()

# Investigate further
# Class Wise



first = train_df['Survived'][train_df["Pclass"] == 1].value_counts()

second = train_df['Survived'][train_df["Pclass"] == 2].value_counts()

third = train_df['Survived'][train_df["Pclass"] == 3].value_counts()



fig, (alt1,alt2, alt3) = plt.subplots(1, 3)



alt1.pie(first, labels=[labels[i] for i in first.keys()], shadow=True, autopct="%1.1f%%")

alt2.pie(second, labels=[labels[i] for i in second.keys()], shadow=True, autopct="%1.1f%%")

alt3.pie(third, labels=[labels[i] for i in third.keys()], shadow=True, autopct="%1.1f%%")



alt1.set_title("First Class")

alt2.set_title("Second Class")

alt3.set_title("Third Class")





alt1.axis('equal')

alt2.axis('equal')

alt3.axis('equal')





plt.show()