class deck:



    def __init__(self, number_of_decks):

        self.number_of_decks = number_of_decks

        self._deck = None

        self.active_deck = None

        self.burnt = None

        self.dealer_final_cut = int()



    def _deck_init(self):



        create = lambda x, y: [x, y]



        self._deck = []

        self.active_deck = {}

        self.burnt = []



        suits = ['Spades', 'Clubs', 'Hearts', 'Diamonds']

        cards = ['Ace', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'Jack', 'Queen', 'King']

        for _x in range(self.number_of_decks):

            for suit in suits:

                for card in cards:

                    self._deck.append(

                        create(suit, card)

                    )



    def shuffle(self):

        self._deck_init()

        np.random.seed(1)

        working_deck = self._deck.copy()

        np.random.shuffle(working_deck)

        np.random.shuffle(working_deck)

        cut1 = np.random.randint(int(len(working_deck) * 0.33), int(len(working_deck) * 0.66))

        working_deck = working_deck[cut1:] + working_deck[:cut1]

        np.random.shuffle(working_deck)

        self.dealer_final_cut = int(len(working_deck) * 0.8)

        for x in range(len(working_deck)):

            self.active_deck[x] = working_deck[x]



    def draw(self):

        keys = list(self.active_deck.keys())

        active_card = [card for card in random.sample(keys, 1) if card not in self.burnt][0]

        self.burnt.append(active_card)

        return self.active_deck[active_card]



    def final(self):

        if len(self.burnt) >= self.dealer_final_cut:

            return True

        else:

            return False
ourdeck = deck(3)

ourdeck.shuffle()

card = ourdeck.draw()

print(card)

print(f"Cards burnt: {ourdeck.burnt}")
class player:

    def __init__(self):

        self.state_single = 0

        self.state_multi = [0, 0]



    def inp(self, card):

        combine_multi = lambda x, y, z: x + y + z



        suit, value = card



        states = {'Ace': [1, 11], 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 'Jack': 10, 'Queen': 10,

                  'King': 10}

        state = states[value]

        if value == 'Ace' or self.state_multi != [0, 0]:

            self.state_multi = [combine_multi(self.state_single, state[0], self.state_multi[0]),

                                combine_multi(self.state_single, state[1], self.state_multi[1])]

        else:

            self.state_single += state

        print(f'{value} of {suit}')



    def oup(self):

        if self.state_multi != [0, 0]:

            print(f'Soft: {self.state_multi[0]}\nHard: (self.state_multi[1])')

        else:

            print(f'Hard: {self.state_single}')



    def reset(self):

        self.state_single = 0

        self.state_multi = [0, 0]
ourdeck = deck(3)

ourplayer = player()



ourdeck.shuffle()



ourplayer.inp(ourdeck.draw())

ourplayer.inp(ourdeck.draw())



ourplayer.oup()

print(f"Cards burnt: {ourdeck.burnt}")