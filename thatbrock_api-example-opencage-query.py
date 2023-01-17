from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("openCageKey")

key = secret_value_0



from opencage.geocoder import OpenCageGeocode

geocoder = OpenCageGeocode(key)

query = 'Seattle, USA'  

results = geocoder.geocode(query)

lat = results[0]['geometry']['lat']

lng = results[0]['geometry']['lng']

print ("Lat: %s, Lon: %s" % (lat, lng))
