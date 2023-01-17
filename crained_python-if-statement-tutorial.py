age = 24

if age >= 20:

    print("You are allowed to join")
age = 19

if age >=20:

    print("You are allowed to join")

else:

    print("Sorry, you cannot join")
age = 18

if age >=20:

    print("You are allowed to join")

elif age == 18:

    print("You can join for an extra $100")

else:

    print("Sorry, you cannot join")
# You can use more than one elif

age = 17

if age >=20:

    print("You are allowed to join")

elif age == 18:

    print("You can join for an extra $100")

elif age == 17:

    print("You can join for an extra $125")

else:

    print("Sorry, you cannot join")
# You can remove the else block just make sure all your logic works

age = 16

if age >=20:

    print("You are allowed to join")

elif age == 18:

    print("You can join for an extra $100")

elif age == 17:

    print("You can join for an extra $125")

elif age < 17:

    print("Sorry, you cannot join")
salad = ['lettuce', 'tomato', 'cucumber', 'carrots', 'cheese']



for salad in salad:

    print(f"Non-vegan salad: {salad}")

    

print("Enjoy your salad")