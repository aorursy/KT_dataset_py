import pandas as pd

import matplotlib.pyplot as plt
train_df = pd.read_csv('../input/train.csv', header=0)
female = train_df.query("Sex == 'female'")

male = train_df.query("Sex == 'male'")



print('Number of Female = ' + str(len(female)))

print('Number of Male   = ' + str(len(male)))
plt.title('Female vs Male (Bar Graph)')

plt.bar(['Female', 'Male'], [len(female), len(male)], color=['mistyrose', 'lightblue'])

plt.show()
plt.title('Female vs Male (Bar Graph w/ Label)')

plt.bar(['Female', 'Male'], [len(female), len(male)], color=['mistyrose', 'lightblue'])

plt.text('Female', len(female), len(female), ha='center', va='bottom')

plt.text('Male', len(male), len(male), ha='center', va='bottom')

plt.show()
female_dead = train_df.query("Sex == 'female' & Survived == 0")

female_survived = train_df.query("Sex == 'female' & Survived == 1")

male_dead = train_df.query("Sex == 'male' & Survived == 0")

male_survived = train_df.query("Sex == 'male' & Survived == 1")



print('Number of Female (Dead)     = ' + str(len(female_dead)))

print('Number of Female (Survived) = ' + str(len(female_survived)))

print('Number of Male (Dead)       = ' + str(len(male_dead)))

print('Number of Male (Survived)   = ' + str(len(male_survived)))
plt.title('Female vs Male (Stacked Bar Graph)')

plt.bar(['Female', 'Male'], [len(female_dead), len(male_dead)], color=['darkred', 'midnightblue'])

plt.bar(['Female', 'Male'], [len(female_survived), len(male_survived)], bottom=[len(female_dead), len(male_dead)], color=['mistyrose', 'lightblue'])

plt.show()
plt.title('Female vs Male (Stacked Bar Graph w/ Label)')

plt.bar(['Female', 'Male'], [len(female_dead), len(male_dead)], color=['darkred', 'midnightblue'])

plt.bar(['Female', 'Male'], [len(female_survived), len(male_survived)], bottom=[len(female_dead), len(male_dead)], color=['mistyrose', 'lightblue'])

plt.text('Female', len(female_dead)/2, "Dead\n" + str(len(female_dead)), color='white', ha='center', va='center')

plt.text('Female', len(female_dead) + len(male_survived)/2, "Survived\n" + str(len(male_survived)), color='black', ha='center', va='center')

plt.text('Male', len(male_dead)/2, "Dead\n" + str(len(male_dead)), color='white', ha='center', va='center')

plt.text('Male', len(male_dead) + len(male_survived)/2, "Survived\n" + str(len(male_survived)), color='black', ha='center', va='center')

plt.show()
test_df = pd.read_csv('../input/test.csv', header=0)
test_df["Survived"] = 0

test_df.loc[test_df["Sex"] == 'female', "Survived"] = 1
test_df = test_df.drop(['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
print(test_df.head(5))

print(test_df.tail(5))
test_df.to_csv(path_or_buf='submission.csv', index=False)