# Priklady zakladnych datovych typov v pythone

# Integer (cele cislo)
print(type(42))

# Float (cislo s plavajucou ciarkov)
print(type(42.42))

# String (retazec)
print(type("Mrkva"))

# List (list)
print(type([1, 4, 6]))

# Touple (Usporiadana n-tica, vektor)
print(type((1, 5, 8)))

# Dictionary (Slovnik - NIKDY tomu nehovorte po slovensky)
print(type({
	"name": "Tony", 
	"surname": "Stark", 
	"nickname": "Ironman"
}))
names = ["Tony", "Bilbo", "Gandalf", "Cartman"]
heights = [1.73, 1.68, 1.94, 1.70]
weights = [56, 65, 56, 120]

bmi = []

for key, height in enumerate(heights):
	bmi.append(weights[key] / (height ** 2))
	
print(bmi)
import numpy as np

names = ["Tony", "Bilbo", "Gandalf", "Cartman"]
heights = [1.73, 1.68, 1.94, 2.01]
weights = [56, 65, 56, 105]

np_heights = np.array(heights)
np_weights = np.array(weights)

bmi = np_weights / (np_heights ** 2)

print(bmi)

print("Maximum: {}".format(bmi.max()))
print("Minimum: {}".format(bmi.min()))
print("Aritmeticky priemer: {}".format(bmi.mean()))
print("Median: {}".format(np.median(bmi)))