
print("What is your name?")
name = input()
print(f"Hello {name}")

questions = [
    "Who \"is\" O'Riely?",
    "What is your favorite color?",
    "What is your favorite book?",
    "What is your favorite animal?"
]

answers = {} #dict

for q in questions:
    print(q)
    a = input()
    answers[q] = a # question is key, answer is value

for k in answers:
    print(k, answers[k])

print("ALL DONE")
print(questions)
print(answers)
# ELIZA AI(Artificial Intelligence) GAME

print("How are you feeling today?")
done = False

while not done: # GAME LOOP - / OPERATING SYSTEM USER INTERFACE "THREAD"
    
    text = input()
    
    text = text.lower()
    
    positive_word_list = [ "ok", "okay", "fine"]
    negative_word_list = [ "bad", "evil", "dumb"]

    if text == "":
        done = True
    elif text in positive_word_list:
        print("Nice! tell me more!")    
    elif text in negative_word_list:
        print("Awww. What are your plans?")    
    elif "?" in text:
        print("Lets talk about you not me. What are your plans?")    
    else:
        print("Interesting; tell me more! (enter blank line to quit)")
        
print("Good bye!")
text = "Now is the time for all good people to come to the aid of their planet."

print("planet" in text)
print("PLANET" in text)

#uppercase
needle = "PLANET"
print(needle.lower() in text)


# If they have milk get a gallon if they have donuts get 12

# JOKE they had donuts so i got 12 gallons

def get(q,kind):
    print(q,kind)

hasMilk = False
hasDonut = True

if hasMilk:
    get(1, "gallon")

if hasDonut:
    get(12, "donuts")
    
from random import randint

qa = [
    "q1",
    "q2",
    "q3",
]

q_index = randint(0, len(qa) - 1)
print(qa[q_index])
