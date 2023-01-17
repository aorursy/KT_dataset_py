Languages = ["Python", "C", "R"]

for x in Languages:

  print(x)
counter = 1

while counter < 6:

  print(counter)

  counter += 1
a = 1

b = 2

if b > a:

  print("b is greater than a")

else :

  print("a is greater than b")
# Use of break statement inside for loop

for val in "string":

    if val == "i":

        break

    print(val)



print("Outside For Loop")
# Use of continue statement inside For loops



for val in "string":

    if val == "i":

        continue

    print(val)



print("Outside For Loop")
#Python Function to reverse a string

def reverse(s): 

  str = "" 

  for i in s: 

    str = i + str

  return str



reverse("WiDSDatathon")
cube = lambda x : x * x * x

print("The cube of 5 is :",cube(5))