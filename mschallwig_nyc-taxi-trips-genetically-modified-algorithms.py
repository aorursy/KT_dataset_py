# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here are several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

import matplotlib.pyplot as plt

import sklearn

from sklearn.neural_network import MLPRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output

#Alright, so let's get started. 

#The first thing we're going to want to do is import our train and test data



trainData = pd.read_csv("../input/train.csv")

testData = pd.read_csv("../input/test.csv")



#We can most likely assume our data is clean, but let's double check to be sure that it's not a big, awful mess



trainColumns = list(trainData.columns)

testColumns = list(testData.columns)

print(set(trainColumns)-set(testColumns))

if len(trainColumns) != len(testColumns):

    print("Not the same number of columns in train and test")

    print("trainColumns has:",len(trainColumns),"values")

    print("testColumns has:",len(testColumns),"values")
#Ok, so we can see that they didn't include the dropoff time and the trip_duration

#Makes sense since finding those is the objective of our ML algorithm in the end anyway, otherwise

#we could just skip ML altogether ^^



#Nice, so now that all of this looks good, let's take a look at our train data in a bit more detail



print(trainData.describe())
#Sometimes not all columns appear - strings and such 

#(notice above we should have 11 columns, but we only see 7 here)

#Let's see the rest of the columns 

print(trainData.columns)
#Ok, so the one's were missing are id, the datetimes for pickup and dropoff, and 

#the store and fwd flag

#let's take a look at the top 5 values of the data, so we can get a feel for how it looks like

print(trainData.head())
#Let's make those id's numeric just so we can use them for plotting

numericIds = []

for numericId in list(trainData["id"]):

    numericIds.append(int(numericId[2:])) #since we're dealing with strings and each string starts

                                        #with id, we just continue from the third character onwards

trainData["numericId"] = numericIds

#and let's take a look to make sure it went well

print(trainData[["id","numericId"]].head())

#And do the same thing for the test data

numericIdsTest = []

for numericIdTest in list(testData["id"]):

    numericIdsTest.append(int(numericIdTest[2:])) #since we're dealing with strings and each string starts

                                        #with id, we just continue from the third character onwards

testData["numericId"] = numericIdsTest

print(testData[["id","numericId"]].head())
#Let's also add in time of the day as well as the date;

#that will be quite useful later if we want to see if

#Trip duration depends on the time of day (morning/evening commutes, school times)

times = []

dates = []

for pickUpDateTime in trainData["pickup_datetime"]:

    time = [int(x) for x in pickUpDateTime.split(" ")[1].split(":")]

    currentDate = [int(x) for x in pickUpDateTime.split(" ")[0].split("-")]

    times.append(time[0]+time[1]/60+time[2]/3600)

    dates.append(datetime.date(currentDate[0],currentDate[1],currentDate[2]))

trainData["pickup_time"] = times

trainData["pickup_date"] = dates

#We should also transform the dates into day of week, because that will also be interesting

#to look at later

dayOfWeek = []

for day in dates:

    dayOfWeek.append(day.weekday())

trainData["pickup_dayOfWeek"] = dayOfWeek

#We'll do the same thing for the testData now

times = []

dates = []

for pickUpDateTime in testData["pickup_datetime"]:

    time = [int(x) for x in pickUpDateTime.split(" ")[1].split(":")]

    currentDate = [int(x) for x in pickUpDateTime.split(" ")[0].split("-")]

    times.append(time[0]+time[1]/60+time[2]/3600)

    dates.append(datetime.date(currentDate[0],currentDate[1],currentDate[2]))

testData["pickup_time"] = times

testData["pickup_date"] = dates

#We should also transform the dates into day of week, because that will also be interesting

#to look at later

dayOfWeek = []

for day in dates:

    dayOfWeek.append(day.weekday())

testData["pickup_dayOfWeek"] = dayOfWeek
#Looks like it went well

#Also, let's add in latitude and longitude changes, as well as distances, so we get a better look

#over of that too, rather than just strange coordinates (??) that aren't intuitive to work with

trainData["latitudeChange"] = trainData["dropoff_latitude"]-trainData["pickup_latitude"]

trainData["longitudeChange"] = trainData["dropoff_longitude"] -trainData["pickup_longitude"]



#To get distance we need to convert latitude and longitude into distance units

latDistance = 111 #1 degree latitude is about 111km

longDistance = 111 #1 degree longitude is also about 111km

trainData["distance"] = np.sqrt(np.power(trainData["latitudeChange"],2)*latDistance+

                               np.power(trainData["longitudeChange"],2)*longDistance)



#And let's just print out the top to make sure everything worked

print(trainData.head())



#And let's do the same for the test data

testData["latitudeChange"] = testData["dropoff_latitude"]-testData["pickup_latitude"]

testData["longitudeChange"] = testData["dropoff_longitude"] -testData["pickup_longitude"]

#Since latDistance and longDistance are equal, we can just take one or the other

testData["distance"] = latDistance*np.sqrt(np.power(testData["latitudeChange"],2)+

                               np.power(testData["longitudeChange"],2))

print(testData.head())
#Nice, looks good so far

#But wowzers, take a look at that number of magnitude in the maximum (and minimum) trip duration

#Let's take a look at the spread of this data

#Good thing we have our numericId now (almost looks like this was planned - WHAT :O)

plt.scatter(trainData["numericId"],trainData["trip_duration"])

plt.xlabel("Unique numeric id")

plt.ylabel("trip duration [s]")

plt.title("Looking for anything suspicious")

plt.show()
#Alright, so we've got some outliers (as suspected)

#Let's take a look at this in two different ways (but way two is much later, just a heads up)

#1.) First we'll isolate the large outliers in time

largeTimesTrain = trainData.loc[trainData["trip_duration"] > 500000].copy()

plt.scatter(largeTimesTrain["numericId"],largeTimesTrain["trip_duration"])

plt.xlabel("Unique numeric id")

plt.ylabel("trip duration [s]")

plt.title("Looking at the outliers")

plt.show()

#Let's also convert these times into minutes and hours

largeTimesTrain["min"] = largeTimesTrain["trip_duration"]/60

largeTimesTrain["hour"] = largeTimesTrain["min"]/60

#It also looks like there aren't a lot of outliers, so we can just print out the whole dataframe

print(largeTimesTrain)
#So, we're not going significant distances (at most a little over 2km <- note:

#we convert the longitude into km distance)

#If we look at the datetimes for pickup and dropoff we see trips lasting between 22 days and a month 

#and 12 days, damn! What??? I've never gotten that lost before

#Our store_and_fwd_flag also tells us that the trip was sent off right after it was completed

#Additionally, although this may just be a conincidence, these trips were all done by the same vendor

#We'll drop these values for now, since there isn't really a logical explanation - just strange

#(at least on first sight) for why the trip was so long, and with about 1.5M data points, there

#really isn't need for speculation



trainData = trainData.loc[trainData["trip_duration"]<500000]

#We'll plot the data one more time now, to see if there is anything else that sticks out

plt.scatter(trainData["numericId"],trainData["trip_duration"])

plt.xlabel("Unique numeric id")

plt.ylabel("trip duration [s]")

plt.title("Looking for anything suspicious")

plt.show()



#Oh, cool, this looks much better

#Now finally, to the second way: looking for outliers on the other hand



#2) Let's plot a histogram this time

#Since we got rid of the big outliers we should have better resolution

#Our first histogram is going to be looking at smaller trip durations

tripDurations = list(trainData["trip_duration"])

plt.hist(tripDurations,bins = range(2000))

plt.xlabel("Trip duration")

plt.ylabel("Frequency")

plt.title("Trip duration histogram for shorter trips")

plt.show()



#The second one will include all trip durations, just in case something big happens to the far right

#I'm gonna leave it commented out though because it takes FOREVER to run

#tripDurations = list(trainData["trip_duration"])

#plt.hist(tripDurations,bins = max(tripDurations))

#plt.xlabel("Trip duration")

#plt.ylabel("Frequency")

#plt.title("Trip duration histogram for all trips")

#plt.show()
#So what we see from the first plot is a nice, right skewed, histogram, 

#with values continuously decreasing when going towards higher trip durations

#We also see that on the left there are a lot of cases around 0, this is probably either because someone

#decided to cancel the fare, or that it was just a test flip, to see if everything is working

#It's unlikely that people take a fare for less than 1-2 minutes unless you're lazy like me

#sometimes you just dont want to walk the 2 mins to the store, you know?



#Let's take a look at how much of our data is concentrated to the far right and how much is

#around 0

print("Number of trips lasting less than 5 hours:",

      len(trainData.loc[trainData["trip_duration"]<18000]["trip_duration"]))

print("Number of trips lasting more than 5 hours:",

      len(trainData.loc[trainData["trip_duration"]>18000]["trip_duration"]))

print("Percentage of trips longer than 5 hours:",

      round(len(trainData.loc[trainData["trip_duration"]>18000]["trip_duration"])/

     len(trainData)*100,2))

print("Number of trips lasting less than 1 minute:",

      len(trainData.loc[trainData["trip_duration"]<60]["trip_duration"]))

print("Number of trips lasting less than 2 minutes:",

      len(trainData.loc[trainData["trip_duration"]<120]["trip_duration"]))

print("Percentage of trips shorter than 1 minute:",

      round(len(trainData.loc[trainData["trip_duration"]<60]["trip_duration"])/

     len(trainData)*100,2))

print("Percentage of trips shorter than 2 minuts:",

      round(len(trainData.loc[trainData["trip_duration"]<120]["trip_duration"])/

     len(trainData)*100,2))

#So about 0.14% of our trips are over 5 hours and about 0.6% of our trips are less than 1 minute

#Let's get a coordinate overview of those cases, maybe there are specific cases this applies to

#If it's randomly scattered, we can probably take it to be insignificant/bad data

#For this we'll create a very short and a long trip duration dataframe

veryshortDurationTrips = trainData.loc[trainData["trip_duration"]<60]

longDurationTrips = trainData.loc[trainData["trip_duration"]>18000].copy()



#We'll scale the point sizes by distance, so we know where long trips occured

plt.scatter(longDurationTrips["pickup_longitude"],longDurationTrips["pickup_latitude"],

            s = 5*longDurationTrips["distance"],alpha = 0.7)

plt.ylabel("pickup latitude")

plt.xlabel("pickup longitude")

plt.show()



#OK, it looks like we have one or some outliers, hard to make out with this resolution

#let's print it out

print(len(longDurationTrips.loc[longDurationTrips["pickup_longitude"]<-75]["distance"]))

print(longDurationTrips.loc[longDurationTrips["pickup_longitude"]<-75]["distance"])
#So we have just one outlier, let's take this out of our consideration and re-do the plot above

cutLongDurationTrips = longDurationTrips.loc[longDurationTrips["pickup_longitude"]>-75]



#We'll scale the point sizes by distance, so we know where long trips occured

plt.scatter(cutLongDurationTrips["pickup_longitude"],cutLongDurationTrips["pickup_latitude"],c="g",

            s = 5*longDurationTrips["distance"],alpha = 0.7)



plt.ylabel("pickup latitude")

plt.xlabel("pickup longitude")



plt.show()
#Now that we know where our data is located, we'll also be plotting all of our trip coordinates,

#so that we can compare

plt.scatter(trainData["pickup_longitude"],trainData["pickup_latitude"],c = "black",

            s = 5*longDurationTrips["distance"],alpha = 0.2,

           label = "all Data") #The alpha lets us compare but also keep

                                                            #the longer durations dominant in 

                                                            #visibility

plt.scatter(cutLongDurationTrips["pickup_longitude"],cutLongDurationTrips["pickup_latitude"],c="g",

            s = 5*longDurationTrips["distance"],alpha = 0.5, label = "long duration")

plt.ylabel("pickup latitude")

plt.xlabel("pickup longitude")



#We'll use the limits seen from the graph above

plt.xlim(-74.05,-73.75)

plt.ylim(40.6,40.9)

plt.legend(loc = "upper right")

plt.show()
#Interesting, so most of the long duration trips are on the main island

#Let's see how things look like when we add in dropoff locations

#Using the same basis as above

plt.scatter(trainData["pickup_longitude"],trainData["pickup_latitude"],c = "black",

            s = 5*longDurationTrips["distance"],alpha = 0.2,

           label = "all Data") #The alpha lets us compare but also keep

                                                            #the longer durations dominant in 

                                                            #visibility

plt.scatter(cutLongDurationTrips["pickup_longitude"],cutLongDurationTrips["pickup_latitude"],c="g",

            s = 5*longDurationTrips["distance"],alpha = 0.5, label = "long dur. pickup")

plt.scatter(cutLongDurationTrips["dropoff_longitude"],cutLongDurationTrips["dropoff_latitude"],c="b",

            s = 5*cutLongDurationTrips["distance"],alpha = 0.5, label = "long dur. dropoff")

plt.ylabel("latitude")

plt.xlabel("longitude")



#We'll use the limits seen from the graph above

plt.xlim(-74.05,-73.75)

plt.ylim(40.6,40.9)

plt.legend(loc = "upper right")

plt.show()
#Ok, so they all start and end on the main island

#What we're really seeing is many small distance trips that have just been counted as very long.



#Let's look at distance one more time using a different lens by

#getting a histogram overview of the distances for long duration trips

#We'll round the distances now, so that we can get at 100m accuracy

longDurationTrips["roundedDistance"] = np.round(longDurationTrips["distance"],1)

plt.hist(longDurationTrips["roundedDistance"],bins = 200)

plt.xlabel("Distance [km]")

plt.ylabel("Frequency")

plt.title("All distances")

plt.show()

longDurationShortDistance = longDurationTrips.loc[longDurationTrips["roundedDistance"]<3].copy()

plt.hist(longDurationShortDistance["roundedDistance"],bins = 30)

plt.xlabel("Distance [km]")

plt.ylabel("Frequency")

plt.title("Short distances")

plt.show()
#Ok, so we see that a lot of long duration trips that take over 5 hours are really

#only over short distances. Even a 20km trip should not take 5 hours unless you drive super duper slow

#So let's filter by trip duration and take everything under 5 hours

#If we take a look at our histogram above, at a trip duration of about 2000 seconds we only have a

#few data points, this just continues to decrease

#ergo, we're going to cut off all of the trip durations over 5 hours, because there are so few

#of them, and it's more likely due to a logging error, or something not concerning

#trip duration than anything else

shortDurationTrips = trainData.loc[trainData["trip_duration"]<18000]
#Let's try to get a better overview of our trip duration

#Maybe it is still somehow related to distance?

plt.scatter(shortDurationTrips["distance"],shortDurationTrips["trip_duration"],s = 2)

plt.xlabel("Trip Distance")

plt.ylabel("Trip Duration")

plt.show()

#Trip duration variation versus time of day

#Trip duration variation versus day of the week
#Ok, it's kinda hard to see a relation here

#Let's cut off the large distance trips, and keep everything under 20km



shortDistancesTrain = shortDurationTrips.loc[trainData["distance"]<20]

plt.scatter(shortDistancesTrain["distance"],shortDistancesTrain["trip_duration"],s = 2)

plt.xlabel("Trip Distance")

plt.ylabel("Trip Duration")

plt.title("Duration vs distance for short distances")

plt.show()
#It looks like once we go slightly over 2.5km, around 3km or so,

#we reach a more predictable range with linearregression, 

#probably because we're leaving the heavy traffic areas

#let's isolate these medium distance cases and take a closer look

mediumDistancesTrain = shortDistancesTrain.loc[shortDistancesTrain["distance"]>3]

plt.scatter(mediumDistancesTrain["distance"],mediumDistancesTrain["trip_duration"],s=10)

plt.xlabel("Trip Distance")

plt.ylabel("Trip Duration")

plt.title("Duration vs distance for medium distances")

plt.show()
#Aright, it still doesn't look super linear, but that's ok, it is a city with traffic and everything

#after all, I wasn't expcecting to do some linear regression, but always worth checking

#Maybe there are some patterns we can see based on the time of day?



plt.scatter(shortDistancesTrain["pickup_time"],shortDistancesTrain["trip_duration"],s = 2)

plt.ylabel("trip duration")

plt.xlabel("pickup time")

plt.title("Time vs duration for all distances")

plt.show()

plt.scatter(mediumDistancesTrain["pickup_time"],mediumDistancesTrain["trip_duration"],s = 10)

plt.ylabel("trip duration")

plt.xlabel("pickup time")

plt.title("Time vs duration for medium distances")

plt.show()

plt.scatter(shortDurationTrips["pickup_time"],shortDurationTrips["trip_duration"],s = 10)

plt.ylabel("trip duration")

plt.xlabel("pickup time")

plt.title("Time vs duration for short durations (<5 hours)")

plt.show()
#That looks pretty messy, let's just plot the average trip duration every minute with its

#standard error of the mean, so we know the range of the mean to expect

fig, ax1 = plt.subplots()

averageTripDurations = []

averageTripDurationsSEM = []

timeOfDay = []

for t in range(12*24):

    currentTime = t/12 #5 minute intervals

    tripsForCurrentTime = shortDurationTrips.loc[shortDurationTrips["pickup_time"] >=currentTime]

    tripsForCurrentTime = tripsForCurrentTime.loc[tripsForCurrentTime["pickup_time"]<currentTime+1/12]

    descriptionForCurrentTime = tripsForCurrentTime["trip_duration"].describe()

    averageTripDurations.append(descriptionForCurrentTime["mean"])

    averageTripDurationsSEM.append(descriptionForCurrentTime["std"]/

                                   descriptionForCurrentTime["count"])

    timeOfDay.append(currentTime)

ax1.errorbar(timeOfDay, averageTripDurations, yerr=averageTripDurationsSEM, fmt='o',ms = 1)

ax1.set_xlabel("Hour of day")

ax1.set_ylabel("Average trip duration")

plt.title("Average trip duration based on time of day")

plt.show()

#Let's look at average trip duration vs time of day 

#(and we can also look at number of trips at each time)
#Alright, that looks pretty cool

#We can't even see the error bars, so that's also a good sign

#To see if there's a relation to how busy things are, let's overlay it with average number of trips

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

numberOfTrips = []

for t in range(12*24):

    currentTime = t/12 #5 minute intervals

    tripsForCurrentTime = shortDurationTrips.loc[shortDurationTrips["pickup_time"] >=currentTime]

    tripsForCurrentTime = tripsForCurrentTime.loc[tripsForCurrentTime["pickup_time"]<currentTime+1/12]

    descriptionForCurrentTime = tripsForCurrentTime["trip_duration"].describe()

    uniqueTripDates = tripsForCurrentTime["pickup_date"].unique()

    numberOfTrips.append(descriptionForCurrentTime["count"]/len(uniqueTripDates))

    

ax1.errorbar(timeOfDay, averageTripDurations, yerr=averageTripDurationsSEM, fmt='o',ms = 1)

ax1.set_xlabel("Hour of day")

ax1.set_ylabel("Average trip duration [s]")

ax2.scatter(timeOfDay,numberOfTrips,color = "r",s = 1,alpha = 0.2)

ax2.set_ylabel("Number of trips per day")

ax2.tick_params(color="r")



plt.title("Average trip duration and number of trips for time of day")

plt.show()

#Ok, interesting, sometimes we see the number of trips a day showing longer trip durations

#Which makes sense, since there is more traffic, but sometimes that case is the opposite

#Let's also add in distance here, maybe that'll explain some things?

#Like a few people taking longer trips to the airport at 5am?

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

ax3 = ax1.twinx()





ax3.spines['right'].set_position(("outward",50))

averageDistance = []

averageDistanceSEM = []

for t in range(12*24):

    currentTime = t/12 #5 minute intervals

    tripsForCurrentTime = shortDurationTrips.loc[shortDurationTrips["pickup_time"] >=currentTime]

    tripsForCurrentTime = tripsForCurrentTime.loc[tripsForCurrentTime["pickup_time"]<currentTime+1/12]

    descriptionForCurrentTime = tripsForCurrentTime["distance"].describe()

    averageDistance.append(descriptionForCurrentTime["mean"])

    averageDistanceSEM.append(descriptionForCurrentTime["std"]/

                                   descriptionForCurrentTime["count"])

    

ax1.errorbar(timeOfDay, averageTripDurations, yerr=averageTripDurationsSEM, fmt='o',ms = 1)

ax1.set_xlabel("Hour of day")

ax1.set_ylabel("Average trip duration [s]")

ax2.scatter(timeOfDay,numberOfTrips,color = "r",s = 1,alpha = 0.2)

ax2.set_ylabel("Number of trips per day")

ax2.tick_params(color="r")

ax3.errorbar(timeOfDay,averageDistance,yerr=averageDistanceSEM,color = "g",ms = 1,alpha = 0.2)

ax3.set_ylabel("Average distance")

ax3.tick_params(color="g")



plt.title("Average trip duration, distance,#trips/day for time of day")

plt.show()
#So we can see, there's a more complicated relationship between distance, number of trips a day, and

#trip duration.

#Sometimes they have an easy relation, other times not soo much

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

ax3 = ax1.twinx()

ax3.spines['right'].set_position(("outward",50))



weekDays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

averageNumberOfTrips = []

averageTripDuration = []

averageTripDurationSEM = []

averageTripDistance = []

averageTripDistanceSEM = []

width = 0.2

for day in range(7):

    tripsForCurrentDay = shortDurationTrips.loc[shortDurationTrips["pickup_dayOfWeek"]==day]

    descriptionForCurrentDate = tripsForCurrentDay.describe()

    uniqueTripDates = tripsForCurrentDay["pickup_date"].unique()

    averageNumberOfTrips.append(descriptionForCurrentDate["trip_duration"]["count"]/

                               len(uniqueTripDates))

    averageTripDuration.append(descriptionForCurrentDate["trip_duration"]["mean"])

    averageTripDurationSEM.append(descriptionForCurrentDate["trip_duration"]["std"]/

                              descriptionForCurrentDate["trip_duration"]["count"])

    averageTripDistance.append(descriptionForCurrentDate["distance"]["mean"])

    distanceSTD = descriptionForCurrentDate["distance"]["std"]

    distanceCount = descriptionForCurrentDate["distance"]["count"]

    averageTripDistanceSEM.append(descriptionForCurrentDate["distance"]["std"]/

                              descriptionForCurrentDate["distance"]["count"])

    

ax1.bar(np.arange(7) - width, averageTripDuration, width, color='r', yerr=averageTripDurationSEM,align = "edge")

ax1.set_ylabel("Average trip duration",color = "red")

ax1.tick_params(color="r")

ax2.bar(np.arange(7), averageNumberOfTrips, width, color='black',align = "edge")

ax2.set_ylabel("Average number of trips")

ax2.tick_params(color="black")

ax3.bar(np.arange(7) + width, np.array(averageTripDistance), width, color='g', yerr=averageTripDistanceSEM,align = "edge")

ax3.set_ylabel("Average trip distance",color = "green")

ax3.tick_params(color="g")

plt.sca(ax1)

plt.xticks(np.arange(7),weekDays,rotation = 45)

plt.show()
#Alright, so we see a little bit of a concave curve happening for each,

#and trips are a bit shorter on weekends than on weekdays, probably because there's less traffic

#So let's go and try to start our ML tasks

#First thing we got to do, split our training data in training and testing

#so that we can check our results



#Preparing the data

finalTrainData = sklearn.utils.shuffle(shortDurationTrips.copy())

dropColumns = ['numericId','id', 'vendor_id','pickup_datetime', 'dropoff_datetime',"pickup_date"]

for extraColumns in dropColumns:

    finalTrainData = finalTrainData.drop(extraColumns, 1)#Getting rid of all the columsn we don't have

                                                        #In test data set

        

#print(finalTrainData.columns)

finalTrainData.replace("N",0,inplace = True)

finalTrainData.replace("Y",1,inplace = True)

dataLength = len(finalTrainData["distance"])#Just getting total number of elements

eightyPerc = int(dataLength*0.8)

restTwenty = dataLength-eightyPerc

finalTrainDataLearn = finalTrainData.head(eightyPerc)

finalTrainDataTarget = finalTrainDataLearn["trip_duration"]

finalTrainDataIndicators = finalTrainDataLearn.drop("trip_duration",1)

finalTrainDataTest = finalTrainData.tail(restTwenty)

finalTrainDataTestTarget = finalTrainDataTest["trip_duration"]

finalTrainDataTestIndicators = finalTrainDataTest.drop("trip_duration",1)

#I'll continue on with the analysis, so make sure to check back soon!