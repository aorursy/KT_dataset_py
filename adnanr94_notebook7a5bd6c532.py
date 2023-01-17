weather_data = []

f = open('../input/la_weather.csv', 'r') # reading file

data = f.read()

rows = data.split('\n')

# making list of lists

for row in rows:

    split_rows = row.split(',')

    weather_data.append(split_rows)

print(weather_data)
weather = []

for row in weather_data:

    b = row[1]

    weather.append(b)

# to remove header

new_weather = weather[1:]

print(new_weather)

count = 0

for row in new_weather:

    count+=1

print(count)
# countig the weather type in a year

weather_counts = {}

for item in new_weather:

    if item in weather_counts:

        weather_counts[item] = weather_counts[item] + 1

    else:

        weather_counts[item] = 1

print(weather_counts)