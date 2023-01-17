import os
import pandas as pd 
ship_dir = '../input'
train_image_dir = os.path.join(ship_dir, 'train')
test_image_dir = os.path.join(ship_dir, 'test')
test_files =[f for f in os.listdir(test_image_dir) ]
submission_df = pd.DataFrame({'ImageId':test_files})
submission_df['EncodedPixels']=None
print (f"There are {len(test_files)} images in the test dataset")
submission_df.head()
submission_df.to_csv('submission.csv',index=False)
