# example of a list below

cars = ['tesla', 'chevy', 'bmw', 'audi']

print(cars)
# We will access 'tesla' from the list

print(cars[0]) #in case it needs to be said. Python always uses a zero to start any list or count
# the second in the list is chevy

print(cars[1])
# if you want to get the last item in a list you use a -1

print(cars[-1])
# We can build on things you have learned already 

print(cars[0].upper())
park = f"there are four {cars[0]}s in the parking lot"

print(park)
teams = ["bills", "browns", "steelers", "bears"]

print(teams)
# now let's add a team to the beginning of the list

teams[0] = "colts"

print(teams)
# you can add to your list with an append

teams.append('bills')

print(teams)
# let's start with an empty list

dogs = []

print(dogs)
dogs.append("poodle")

dogs.append("bulldog")

print(dogs)
dogs.insert(1, "collie")

print(dogs)
del dogs[1]

print(dogs)
# you can also remove by value

dogs.remove("poodle")
print(dogs)
# now we can remove and us it in a different way

best_team = "bills"

teams.remove(best_team)

print(teams)
print(f"The best team is the {best_team}")
active_users = ["jim", "chris", "sara"]

print(active_users)
# now let's say we want to pop the last user off the list

inactive_users = active_users.pop()

print(active_users)
print(inactive_users)
active_users2 = ["jim", "chris", "sara"]

print(active_users2)
# we want to make chris inactive

inactive_users2 = active_users2.pop(1)

print(active_users2)
print(inactive_users2)
# we can sort by alphabetical

teams.sort()

print(teams)
# we can reverse this

teams.sort(reverse=True)

print(teams)
print("Good team list:")

print(teams)
# sort the list by alphabetical

print("Good team list:")

print(sorted(teams))
len(teams)