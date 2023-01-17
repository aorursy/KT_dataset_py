# Question 1
# age: 60 
# workclass: Private
# marital-status: Divorced 
# race: Black
# sex: Female


import pandas as pd

adult_data = pd.read_csv('../input/toludata/adult.data.csv')
columns = adult_data.columns

# Convert to string
adult_data = adult_data.astype(str)

# Remove white space
adult_data = adult_data.apply(lambda x: x.str.strip())
print(adult_data.columns)

# Question 1a

rows_match = adult_data.loc[(adult_data['age'] == '60') & (adult_data['workclass'] == 'Private') & (adult_data['marital-status'] == 'Divorced') & (adult_data['race'] == 'Black') & (adult_data['sex'] == 'Female')]
print(rows_match)
# Question 1b

num_of_match = rows_match.shape[0]
print(num_of_match)

# Question 1c

rows_match_with_education = rows_match.loc[(rows_match['education'] == '10th')]
num_of_match2 = rows_match_with_education.shape[0]
print(num_of_match2)
# Question 1d

print(rows_match_with_education)

# Question 2

bob = pd.read_csv('../input/bobdata/bob.csv')
purchase = pd.read_csv('../input/toludata/purchase-rest.csv')

bob_row = bob.iloc[0].iloc[1:].astype(int)
#print(bob_row)

# Question 2a

def get_indices(from_list):
    indices = [index for index, val in enumerate(from_list) if val == 1]
    return set(indices)

print(get_indices([1,0,1,1,0]))
# Question 2b
# Compute jaccard index

from sklearn.metrics import jaccard_score

def compute_jaccard_index(y_true, y_pred):
    return jaccard_score(y_true, y_pred)
    

# Question 2c

jaccard_scores = pd.DataFrame(columns=['indx','user', 'score'])
    

for index, row in purchase.iterrows():
    temp_jacc = compute_jaccard_index(bob_row, row.iloc[1:])
    jaccard_scores.loc[index] = [index, row.iloc[0], temp_jacc]
    #jaccard_scores.append({'idx': index, 'user': row.iloc[0], 'score': temp_jacc}, ignore_index=True)


top_two = jaccard_scores.nlargest(2, 'score')

# convert user and indx field to integer
top_two['indx'] = top_two['indx'].astype(int)
top_two['user'] = top_two['user'].astype(int)
print(top_two)
print(top_two['user'])
# Question 2d

details = top_two.merge(purchase, on='user', how='left')
print(details)

