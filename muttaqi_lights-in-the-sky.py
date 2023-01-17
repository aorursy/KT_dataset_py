# The data in this analysis comes from a dataset downloaded from the Kaggle website about

# historical UFO sightings.



# I am new to data analytics and to Python. So this was just a simple project

# for me to get comfortable with analyzing data.



# My goal was to find some interesting information about these UFO sightings. What follows comes from

# five hours of playing around with the original dataset and with python.

import pandas as pd

%matplotlib inline
# There were a lot of problems with the original dataset and required some cleaning. That probably could

# have been done in Python, but I am more familiar with Excel. Nonetheless, it took a while for me

# to get the dataset somewhat usable.

ufo = pd.read_csv('../input/ufoclean.csv', low_memory = False)

ufo.info()
# The first thing was to find the most common shape of UFO sightings. UFO sightings are often described 

# in many different ways. According to this dataset, they are most often reported as lights in the sky.

shape_counts = ufo['Shape'].value_counts()

shape_counts.head
# This is a bar graph showing the 10 most common shapes that UFO sightings take. Lights are reported

# almost twice as much as the second most common shape.

shape_counts[:10].plot(kind='bar')
# Since lights were the most common shape, we'll take a closer look at them

Light = ufo['Shape'] == 'light'

streaks = ufo[Light]
# Now to find out how often each state reports lights as UFO sightings.

streak_counts = streaks['State'].value_counts()

streak_counts[:20]
# California appears to report lights in the sky as UFO's more often than any other state.

# This is not surprising since California also reports more UFO sightings than any other state.

streak_counts[:10].plot(kind='bar')
# Now to see on which days are lights most often seen as UFO's

# The first line is to change the Date format in the datafram to something Pandas can use

# The second line is to convert that date format to days of the week

# 0 = Monday, 1 = Tuesday, and so on...

ufo['day'] = pd.to_datetime(ufo['Date'])

ufo['weekday'] = ufo['day'].dt.dayofweek



# This is to find on which days UFO's are often seen and reported as lights

weekday_lights = Light.groupby(ufo['weekday']).aggregate(sum)

print(weekday_lights)
# This is to convert the days of the week from numbers to their regular, English words

# As we can see, lights in the sky are reported as UFO's more on Saturdays and Sundays than on other

# days of the week.

weekday_lights.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

weekday_lights.plot(kind='bar')