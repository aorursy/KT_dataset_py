import os
dataDir = '/kaggle/input/prostate-cancer-grade-assessment/train_images'

dataTestDir = '/kaggle/input/prostate-cancer-grade-assessment/test_images'
print(os.path.exists(dataDir))
print(os.path.exists(dataTestDir))