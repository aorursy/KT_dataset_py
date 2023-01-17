import json



with open('../input/usa-dunkin-donuts-stores/dunkinDonuts.json') as file:

    data = json.load(file)["data"]



latitSum = 0

longitSum = 0

for entry in data:

    latitSum += entry["geoJson"]["coordinates"][0]

    longitSum += entry["geoJson"]["coordinates"][1]

    

avgLat = latitSum / len(data)

avgLong = longitSum / len(data)

print(avgLat, avgLong)

print("The average USA Dunkin Donuts is somewhere in West Virginia")