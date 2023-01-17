import pandas as pd
df = pd.read_csv('../input/fifa19/data.csv')
right_filt = (df['Preferred Foot'] == 'Right')
left_filt = (df['Preferred Foot'] == 'Left')
not_other_filt = (df['Preferred Foot'].isin(['Right', 'Left']))
right = df[right_filt]
left = df[left_filt]
other = df[~not_other_filt]
# Number of right-footed players:
len(right.index)
# Number of left-footed players:
len(left.index)
# Number of players for whom no footedness information has been given:
len(other.index)
