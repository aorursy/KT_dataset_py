import pandas as pd
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

submission = pd.read_csv("../input/sampleSubmission.csv")
print("Train dataset:")

print(train.head())

print("Test dataset:")

print(test.head())

print("Sample submission:")

print(submission.head())
print(train.describe())
mean = train.describe()["count"]["mean"]
submission["count"] = mean
submission.to_csv("submission.csv", index=False)
import pip

installed_packages = pip.get_installed_distributions()

installed_packages_list = sorted(["%s==%s" % (i.key, i.version)

     for i in installed_packages])

print(installed_packages_list)