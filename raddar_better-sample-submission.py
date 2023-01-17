import pandas as pd



sub = pd.read_csv('../input/sample_submission.csv')



tmp = sub.groupby('ImageId')['ImageId'].count().reset_index(name='N')

tmp = tmp.loc[tmp.N > 1] #find image id's with more than 1 row -> has pneumothorax mask!

sub.loc[sub.ImageId.isin(tmp.ImageId),'EncodedPixels'] = f"1 {1024*1024}"



sub.to_csv('sample_submission2.csv',index=None)