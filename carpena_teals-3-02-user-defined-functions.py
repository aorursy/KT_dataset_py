# =========================
# Unabstracted Code
# =========================

print ("### GET Ready to Play TIC-TAC-TOE!!! ### \nA Mr. Oseguera Python Class production\n\n")
v = '|  *  |  *  |  *  |'
h = ' _____ _____ _____ '
print(h)
print(v)
print(h)
print(v)
print(h)
print(v)
print(h)
def draw_tictactoe_board():
    print ("### GET Ready to Play TIC-TAC-TOE!!! ### \nA Mr. Oseguera Python Class production\n\n")
    v = '|  *  |  *  |  *  |'
    h = ' _____ _____ _____ '
    i = 0
    while i <= 6:
        if i % 2 == 0: #modulus
            print(h)
        else:
            print(v)
        i += 1
# +++++++++++++++++++++++++
# Abstracted Code
# +++++++++++++++++++++++++

draw_tictactoe_board()
import random  #include features and functionalities of this library in your local code
card_deck = ['Ace of Spades', 'King of Diamonds', '3 of Hearts' , '10 of Clubs', 'Queen of Diamonds'] #instantiates your cards in a list
#random.choice
random.choice(card_deck) #asks python to pick an item in your list using the random 
#random.shuffle
random.shuffle(card_deck)
card_deck[:]