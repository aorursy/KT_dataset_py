import pandas as pd



train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

train_labels.groupby(['title'])['accuracy_group'].mean()
scores_dict = {

    'Bird Measurer (Assessment)': 1,

    'Cart Balancer (Assessment)': 2,

    'Cauldron Filler (Assessment)': 2,

    'Chest Sorter (Assessment)': 1,

    'Mushroom Sorter (Assessment)': 2

}
test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')

test_users = set(test['installation_id'])
%%time

sub = []

for u in test_users:

    user_data = test.loc[test['installation_id']==u, :] 

    user_data = user_data.sort_values(['timestamp']) # sort by time

    assessment = list(user_data['title'])[-1] # title of the last event (always an assessment)

    score = scores_dict[assessment] # score the assesment

    sub.append([u, score])
sub_df = pd.DataFrame(sub, columns=['installation_id', 'accuracy_group'])

sub_df.to_csv('submission.csv', index=False)