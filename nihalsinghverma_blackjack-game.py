import random as rd

#from art import tprint



"""

Gobal variables

"""

suits   = ('Hearts','Diamonds','Spades','Clubs')

ranks   = ('Ace','Two','Three','Four','Five','Six','Seven','Eight','Nine','Ten','Jack','Queen','King')

values  = {'Two':2, 'Three':3, 'Four':4, 'Five':5, 'Six':6, 'Seven':7, 'Eight':8, 'Nine':9, 'Ten':10, 'Jack':10, 'Queen':10, 'King':10, 'Ace':11}

playing = True

mchips  = 0



print("\n===========================================================================================================================")

print("Welcome to BlackJack") #tprint("Welcome to BlackJack")

print("===========================================================================================================================")

name    = input("\nYou have total of 1000 chips in your a/c for bet.\nNow please enter your name:").title()





class Card(object):	

	"""

	A Card object really only needs two attributes: suit and rank.

	You might add an attribute for "value" - we chose to handle value later when developing our Hand class.

	In addition to the Card's __init__ method, consider adding a __str__ method that, 

	when asked to print a Card, returns a string in the form "Two of Hearts"

	"""

	def __init__(self,suit,rank):

		self.suit = suit

		self.rank = rank



	def __str__(self):

		return self.rank + " of " + self.suit



class Deck(object):	

	"""

	Here we might store 52 card objects in a list that can later be shuffled.

	First, though, we need to *instantiate* all 52 unique card objects and add them to our list.

	So long as the Card class definition appears in our code, we can build Card objects inside our Deck __init__ method.

	Consider iterating over sequences of suits and ranks to build out each card. This might appear inside a Deck class.

	"""	

	def __init__(self):



		self.deck = [] # start with the empty list. 

		for suit in suits:

			for rank in ranks:

				self.deck.append(Card(suit,rank)) # build Card objects and add them to the list



	def __str__(self):

		

		deck_comp = ""

		for card in self.deck:

			deck_comp += "\n" + card.__str__() # add each Card object's print string

		return "The deck has: "+deck_comp



	def shuffle(self):

		rd.shuffle(self.deck)



	def deal(self):

		single_card = self.deck.pop()

		return single_card



class Hand(object):

	"""

	In addition to holding Card objects dealt from the Deck, 

	the Hand class may be used to calculate the value of those cards using the values dictionary defined above. 

	It may also need to adjust for the value of Aces when appropriate.

	"""

	def __init__(self):

		self.cards = []  # start with an empty list as we did in the Deck class\n",

		self.value = 0   # start with zero value\n",

		self.aces = 0    # add an attribute to keep track of aces\n",



	def add_card(self,card):

		self.cards.append(card)

		self.value += values[card.rank]

		if card.rank == 'Ace':

			self.aces += 1



	def adjust_for_ace(self):

		while self.value > 21 and self.aces:

			self.value -= 10

			self.aces -= 1



class Chips(object):

	"""

	In addition to decks of cards and hands, we need to keep track of a Player's starting chips, bets, and ongoing winnings.

	This could be done using global variables, but in the spirit of object oriented programming, let's use a Chips class instead!

	"""

	def __init__(self,total=100):

		self.total = total # This can be set to a default value or supplied by a user input

		self.bet = 0



	def win_bet(self):

		self.total += self.bet



	def lose_bet(self):

		self.total -= self.bet



def take_bet(chips):



	while True:

		try:

			chips.bet = int(input("How many chips would you like to bet: "))

		except ValueError:

			print("Sorry, Bet must be an integer!")

		else:

			if chips.bet > chips.total:

				print("Sorry, You don't have enough chips to bet!")

				choice = input("You wanna buy some chips for bet? Press 'Y' or 'N': ")

				if choice.upper() == 'Y':

					while True:

						try:

							bchips = int(input("How many chips you wanna buy for bet: "))

						except ValueError:

							print("Sorry, Chips amount must be an integer!")

						else:

							chips.total += bchips

							print("You have total "+str(chips.total)+" in your hand.")

							#print("Now you can bet.")

							break

			else:

				break



def buy_chips():

	pass



def hit(deck,hand):

	hand.add_card(deck.deal())

	hand.adjust_for_ace()



def hit_or_stand(deck,hand):

	global playing # to control an upcoming while loop

	while True:

		x = input("Would you like to Hit or Stand? Enter 'h' or 's': ")



		if x[0].upper() == 'H':

			hit(deck,hand)



		elif x[0].upper() == 'S':

			print(name+" stands. Dealer is playing.")

			playing = False



		else:

			print("Sorry, please try again...")

			continue

		break



def show_some(player,dealer):

    print("\nDealer's Hand:")

    print(" <card hidden>")

    print('',dealer.cards[1])  

    print("\n"+name+"'s Hand:", *player.cards, sep='\n ')

    

def show_all(player,dealer):

    print("\nDealer's Hand:", *dealer.cards, sep='\n ')

    print("Dealer's Hand =",dealer.value)

    print("\n"+name+"'s Hand:", *player.cards, sep='\n ')

    print("\n"+name+"'s Hand =",player.value)



def player_busts(player,dealer,chips):

    print(name+", busts!")

    chips.lose_bet()



def player_wins(player,dealer,chips):

    print(name+", You wins!")

    chips.win_bet()



def dealer_busts(player,dealer,chips):

    print("Dealer busts!")

    chips.win_bet()

    

def dealer_wins(player,dealer,chips):

    print("Dealer wins!")

    chips.lose_bet()

    

def push(player,dealer):

    print("Dealer and Player tie! It's a push.")



while True:

	# Print an opening statement

	print('\nHi '+name+'!Get as close to 21 as you can without going over! Dealer hits until he/she reaches 17. Aces count as 1 or 11.\n')



	# Create & shuffle the deck, deal two cards to each player

	deck = Deck()

	deck.shuffle()



	player_hand = Hand()

	player_hand.add_card(deck.deal())

	player_hand.add_card(deck.deal())



	dealer_hand = Hand()

	dealer_hand.add_card(deck.deal())

	dealer_hand.add_card(deck.deal())

	        

	# Set up the Player's chips

	player_chips = Chips(mchips)  # remember the default value is 1000    



	# Prompt the Player for their bet

	take_bet(player_chips)



	# Show cards (but keep one dealer card hidden)

	show_some(player_hand,dealer_hand)



	while playing:  # recall this variable from our hit_or_stand function

	    

	    # Prompt for Player to Hit or Stand

	    hit_or_stand(deck,player_hand) 

	    

	    # Show cards (but keep one dealer card hidden)

	    show_some(player_hand,dealer_hand)  

	    

	    # If player's hand exceeds 21, run player_busts() and break out of loop

	    if player_hand.value > 21:

	        player_busts(player_hand,dealer_hand,player_chips)

	        break        





	# If Player hasn't busted, play Dealer's hand until Dealer reaches 17 

	if player_hand.value <= 21:

	    

	    while dealer_hand.value < 17:

	        hit(deck,dealer_hand)    



	    # Show all cards

	    show_all(player_hand,dealer_hand)

	    

	    # Run different winning scenarios

	    if dealer_hand.value > 21:

	        dealer_busts(player_hand,dealer_hand,player_chips)



	    elif dealer_hand.value > player_hand.value:

	        dealer_wins(player_hand,dealer_hand,player_chips)



	    elif dealer_hand.value < player_hand.value:

	        player_wins(player_hand,dealer_hand,player_chips)



	    else:

	        push(player_hand,dealer_hand)        



	# Inform Player of their chips total 

	print("\n"+name+" stand at",player_chips.total)

	mchips = player_chips.total



	# Ask to play again

	new_game = input("\nWould you like to play another hand? Enter 'y' or 'n': ")



	if new_game[0].lower()=='y':

	    playing=True

	    continue

	else:

	    print("Thanks "+name+" for playing!")

	    break