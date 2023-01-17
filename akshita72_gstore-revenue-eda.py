import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

trainData = pd.read_csv("../input/train.csv")
testData = pd.read_csv("../input/test.csv")

import json
from pandas.io.json import json_normalize

cols = ["device","geoNetwork","totals","trafficSource"]

for col in cols:
    trainData = trainData.join(json_normalize(
        trainData[col].map(json.loads).tolist())
                               .add_prefix(col+".")).drop([col], axis=1)
    testData = testData.join(json_normalize(
        testData[col].map(json.loads).tolist())
                               .add_prefix(col+".")).drop([col], axis=1)

per = (trainData.isnull().sum()/trainData.shape[0])*100
percents = per.iloc[per.nonzero()[0]]

from matplotlib import pyplot as plt
percents.plot.barh()
plt.show()
plt.figure()
plt.tight_layout()
plt.subplots_adjust(left=49, bottom=49, right=50, top=50,
                wspace=0.5, hspace=0.5)

plt.subplot(221)
trainData["device.browser"].value_counts().nlargest(10).plot("barh",width = 1).invert_yaxis()
plt.title("Browser")
plt.xlabel('Count')
plt.ylabel('Browser')

plt.subplot(222)
trainData["device.deviceCategory"].value_counts().plot("barh",width = 1).invert_yaxis()
plt.title("DeviceCategory")
plt.xlabel('Count')
plt.ylabel('Category')

plt.subplot(223)
trainData["device.operatingSystem"].value_counts().nlargest(10).plot("barh",width = 1).invert_yaxis()
plt.title("OperatingSystem")
plt.xlabel('Count')
plt.ylabel('Operating System')

plt.subplot(224)
trainData["device.isMobile"].value_counts().plot("barh",width = 1).invert_yaxis()
plt.title("Device is Mobile or Not")
plt.xlabel('Count')
plt.ylabel('Is Mobile or Not?')

plt.show()


trainData = trainData.drop(["device.browserSize","device.browserVersion","device.flashVersion",
               "device.language","device.mobileDeviceBranding","device.mobileDeviceInfo",
               "device.mobileDeviceMarketingName","device.mobileDeviceModel","device.mobileInputSelector",
               "device.operatingSystemVersion","device.screenColors","device.screenResolution"],axis=1)
testData = testData.drop(["device.browserSize","device.browserVersion","device.flashVersion",
               "device.language","device.mobileDeviceBranding","device.mobileDeviceInfo",
               "device.mobileDeviceMarketingName","device.mobileDeviceModel","device.mobileInputSelector",
               "device.operatingSystemVersion","device.screenColors","device.screenResolution"],axis=1)
plt.figure()
plt.tight_layout()
plt.subplots_adjust(left=49, bottom=49, right=50, top=50,
                wspace=0.5, hspace=1)

plt.subplot(311)
trainData["geoNetwork.continent"].value_counts().plot("barh",width = 1).invert_yaxis()
plt.title("Continent")
plt.xlabel('Count')
plt.ylabel('Continent')

plt.subplot(312)
trainData["geoNetwork.subContinent"].value_counts().nlargest(10).plot("barh",width = 1).invert_yaxis()
plt.title("Sub-Continent")
plt.xlabel('Count')
plt.ylabel('subContinent')

plt.subplot(313)
trainData["geoNetwork.country"].value_counts().nlargest(10).plot("barh",width = 1).invert_yaxis()
plt.title("Country")
plt.xlabel('Count')
plt.ylabel('country')

plt.show()

trainData = trainData.drop(["geoNetwork.city","geoNetwork.cityId","geoNetwork.latitude","geoNetwork.longitude",
                "geoNetwork.metro","geoNetwork.networkDomain","geoNetwork.networkLocation",
                "geoNetwork.region"],axis=1)
testData = testData.drop(["geoNetwork.city","geoNetwork.cityId","geoNetwork.latitude","geoNetwork.longitude",
                "geoNetwork.metro","geoNetwork.networkDomain","geoNetwork.networkLocation",
                "geoNetwork.region"],axis=1)
trainData.dtypes
trainData["totals.bounces"] = trainData["totals.bounces"].astype(np.float64)
trainData["totals.hits"] = trainData["totals.hits"].astype(np.float64)
trainData["totals.newVisits"] = trainData["totals.newVisits"].astype(np.float64)
trainData["totals.pageviews"] = trainData["totals.pageviews"].astype(np.float64)
trainData["totals.transactionRevenue"] = trainData["totals.transactionRevenue"].astype(np.float64)
trainData["totals.visits"] = trainData["totals.visits"].astype(np.float64)
testData["totals.bounces"] = testData["totals.bounces"].astype(np.float64)
testData["totals.hits"] = testData["totals.hits"].astype(np.float64)
testData["totals.newVisits"] = testData["totals.newVisits"].astype(np.float64)
testData["totals.pageviews"] = testData["totals.pageviews"].astype(np.float64)
testData["totals.visits"] = testData["totals.visits"].astype(np.float64)
print(trainData["totals.bounces"].isnull().sum())
print(trainData["totals.bounces"].unique())
trainData["totals.bounces"].fillna(0, inplace=True)
testData["totals.bounces"].fillna(0, inplace=True)

print(trainData["totals.hits"].isnull().sum())
print(trainData["totals.hits"].unique())

print(testData["totals.hits"].isnull().sum())
print(testData["totals.hits"].unique())
print(trainData["totals.newVisits"].isnull().sum())
trainData["totals.newVisits"].unique()

trainData["totals.newVisits"].fillna(0, inplace=True)
testData["totals.newVisits"].fillna(0, inplace=True)

print(trainData["totals.newVisits"].unique())
print(testData["totals.newVisits"].unique())
print(trainData["totals.pageviews"].isnull().sum())
print(trainData["totals.pageviews"].unique())

trainData["totals.pageviews"].fillna(0, inplace=True)
testData["totals.pageviews"].fillna(0, inplace=True)

print(trainData["totals.pageviews"].unique())
print(testData["totals.pageviews"].unique())
print(trainData["totals.visits"].isnull().sum())
trainData["totals.visits"].unique()


trainData = trainData.drop(["totals.visits"],axis=1)
testData = testData.drop(["totals.visits"],axis=1)
print(trainData["totals.transactionRevenue"].isnull().sum())
print(trainData["totals.transactionRevenue"].unique())

trainData["totals.transactionRevenue"].fillna(1, inplace=True)
print(trainData["totals.transactionRevenue"].unique())
trainData.dtypes
trainData["trafficSource.adContent"].value_counts().nlargest(10).plot("barh",width = 1).invert_yaxis()
#plt.title("Continent")
#plt.xlabel('Count')
#plt.ylabel('Continent')
trainData["trafficSource.adwordsClickInfo.adNetworkType"].value_counts().nlargest(10).plot("barh",width = 1).invert_yaxis()

trainData["trafficSource.adwordsClickInfo.criteriaParameters"].value_counts().nlargest(10).plot("barh",width = 1).invert_yaxis()

trainData["trafficSource.adwordsClickInfo.gclId"].value_counts().nlargest(10).plot("barh",width = 1).invert_yaxis()

trainData["trafficSource.adwordsClickInfo.isVideoAd"].value_counts().nlargest(10).plot("barh",width = 1).invert_yaxis()

trainData["trafficSource.adwordsClickInfo.page"].value_counts().nlargest(10).plot("barh",width = 1).invert_yaxis()

trainData["trafficSource.adwordsClickInfo.slot"].value_counts().nlargest(10).plot("barh",width = 1).invert_yaxis()
