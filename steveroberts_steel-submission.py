import pandas as pd



# s = pd.read_csv('../input/severstal-fast-ai-256x256-crops-sub/submission.csv')

# s = pd.read_csv('../input/colab1/catalyst_submission (1).csv')

# s = pd.read_csv('../input/colab1/catalyst_submission_2segmenters.csv')

# s = pd.read_csv('../input/colab1/submission_3classifiers_3segmenters.csv')

# s = pd.read_csv('../input/colab1/pspnet_submission.csv')



# s = pd.read_csv('../input/colab1/submission_resnet18_Unet_fold0.csv')

# s = pd.read_csv('../input/colab1/submission_resnet50_fpn_fold0.csv')



s = pd.read_csv('../input/colab1/submission_3classifiers_11segmenters.csv')



s.to_csv('submission.csv',index=False)

s.head()
s.shape