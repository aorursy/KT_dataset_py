a = 10
b = 5
print(a+b)
# You can reference variables between cells
print(a/b)
h = "hello"
e = "world"
print(h + " " + e)
text = h + " " + e
print(text)
# Once defined, most variable types have methods that can be called.
# For example, string variables have .title()
print(text.title())
# Lists
dogs = ['border collie', 'beagle', 'labrador retriever']
print(dogs)
dog = dogs[0]
print(dog.title())
# Lists and looping
for dog in dogs:
    print(dog)
# Modifying a list
dogs[0] = 'huskie'
dogs.append('basset hound')
print(dogs)
print('chihuahua' in dogs)
# Tuples: Lists that can't be changed
cats = ('russian blue', 'tortoise shell')
print(cats[1])
cats.append('tabby')
# Can just perform an activity
def thank_you(name):
    print("You are doing very good work, %s!" % name)
    
thank_you('India')
thank_you('Terror')
thank_you('Disillusionment')
# Or can return a value
def add_numbers(x,y):
    return x + y
# You can use values
print(add_numbers(666, 333))
# Or variables
print(add_numbers(a,b))
print(5 == 5)
print(3 == 5)
print(a == b)
print('a' == 'b')
print('Andy'.lower() == 'andy'.lower())
print(a != b)
print(a < b)
import this
# A simple map
import folium
m = folium.Map(location=[44.05, -121.3])
m
# Add a marker
m2 = folium.Map(
    location=[44.05, -121.3],
    zoom_start=12,
    tiles='Stamen Terrain'
)
folium.Marker([44.042, -121.333], popup='Here we are!').add_to(m2)
m2
# Get our geojson
import requests
import json
with open('../input/ne_110m_rivers_lake_centerlines.geojson') as geojson:
    rivers = json.load(geojson)

m3 = folium.Map(
    tiles='Mapbox Bright'
)

folium.GeoJson(
    rivers,
    name='Rivers'
).add_to(m3)

m3
