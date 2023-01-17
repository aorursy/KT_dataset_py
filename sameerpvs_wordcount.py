import pandas as pd
train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

gt_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

#gt_df = pd.read_csv("../input/disasters-on-social-media/socialmedia-disaster-tweets-DFE.csv")
gt_df = gt_df[['id','text']]



gt_df['word_count'] = gt_df['text'].apply(lambda x: len(str(x).split()))



sum =0

count =0

for word in gt_df['word_count']:

    sum = sum + int(word)

    count = count +1

rem = int(sum/count)

print(rem)

Type_new = pd.Series([]) 

for i in range(len(gt_df)): 

    if gt_df["word_count"][i] > rem: 

        Type_new[i]=1

  

    else: 

        Type_new[i]= 0 

  

          

# inserting new column with values of list made above         

gt_df.insert(3,"target", Type_new) 

gt_df['id'] = gt_df.id

gt_df = gt_df[['id','target']]

gt_df
subm_df = gt_df[['id', 'target']]

subm_df
subm_df.to_csv('submission.csv', index=False)