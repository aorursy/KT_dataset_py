from random import randint


class Character:
    def __init__(self, strength=9, dexterity=9, health=100, name="character", description="a game character"):
        self.strength = strength
        self.dexterity = dexterity
        self.name = name
        self.description = description
        self.health = health
        self.inventory = []
        self.location = None

    def __str__(self):
        return f"NAME: {self.name}, HEALTH: {self.health}"

    def collect(self, item):
        if item.kind == "HEALTH" and (100 - self.health) >= item.health_bonus_percent:
            self.health += item.health_bonus_percent
        else:
            self.inventory.append(item)

    def display_inventory(self):
        print("INVENTORY:")
        for item in self.inventory:
            print(f"\t{item}")

    def attack(self, opponent):
        limit = self.strength
        number = randint(1, limit) + randint(1, limit) + randint(1, limit)
        number -= randint(1, opponent.dexterity)
        opponent.health -= number
        print(f"{self.name} does {number} damage to {opponent.name}")


class Ogre(Character):
    def __init__(self):
        super().__init__()
        self.name = "Ogre"
        self.health = 50


class Player(Character):
    pass


class InventoryItem:
    def __init__(self, name, kind="Treasure"):
        self.name = name
        self.kind = kind

    def __str__(self):
        return f"{self.name} {self.kind}"


class Gold(InventoryItem):
    def __init__(self, quantity):
        super().__init__("gold")
        self.quantity = quantity

    def __str__(self):
        return f"{self.quantity} {self.name} pieces."


class FirstAidKit(InventoryItem):
    def __init__(self, health_bonus_percent=25):
        super().__init__("FirstAidKit", "HEALTH")
        self.health_bonus_percent = health_bonus_percent


class Weapon(InventoryItem):
    def __init__(self, name, max_damage):
        super().__init__(name, "WEAPON")
        self.max_damage = max_damage


class Sword(Weapon):
    def __init__(self):
        super().__init__("Sword", "WEAPON")
        self.max_damage = 20


# rows = [
#     ['.', '.', '.', ],
#     ['.', '.', '.', ],
#     ['.', '.', '.', ],
# ]


class Board:
    def __init__(self, size=9):
        self.rows = []
        for row_index in range(size):
            values = []
            for value_index in range(size):
                values.append(".")
            self.rows.append(values)

    def render(self):

        for row in self.rows:
            for value in row:
                print(value, end="  ")
            print("")

    def place(self, location, symbol="."):
        self.rows[location.row-1][location.column-1] = symbol


class Location:
    def __init__(self, row, column):
        self.row = row
        self.column = column


class Game:
    def __init__(self):
        self.player = Player(name="Kevin")
        self.player.dexterity = 18
        self.size = 9
        self.playing = True
        self.level = 1
        self.reset()

    def random_location(self):
        return Location(randint(0, self.size - 1), randint(0, self.size - 1))

    def reset(self):
        self.player.location = self.random_location()
        self.monsters = []
        quantity = self.level * 2
        for q in range(quantity):
            ogre = Ogre()
            for i in range(3): #GIVE MONSETER THREE ITEMS
                self.add_random_item(ogre)
            ogre.location = self.random_location()
            self.monsters.append(ogre)

    def play(self):
        while self.playing:
            self.display()
            self.apply_user_input()
        print("G A M E   O V E R")

    def apply_user_input(self):
        option_list = {
            "0": "QUIT",
            "1": "NORTH",
            "2": "EAST",
            "3": "SOUTH",
            "4": "WEST",
        }

        choice = self.get_choice(option_list)

        if choice == "QUIT":
            self.playing = False
        elif choice in ["NORTH","EAST","SOUTH","WEST"]:
            self.move_player(choice)
        else:
            print("INVALID OPTION")

        self.check_collisions()

    def check_collisions(self):
        for m in self.monsters:
            if self.player.location.row == m.location.row and self.player.location.column == m.location.column:
                self.fight(m)
                if m.health <= 0:
                    for item in m.inventory: #TAKE THE MONSTERS STUFF
                        self.player.collect(item)
                    self.monsters.remove(m)

        if len(self.monsters) == 0:
            self.level += 1
            self.reset()

    def move_player(self, direction):
        if direction == "NORTH":
            self.player.location.row -= 1
        elif direction == "SOUTH":
            self.player.location.row += 1
        elif direction == "EAST":
            self.player.location.column += 1
        elif direction == "WEST":
            self.player.location.column -= 1

    def display(self):
        print(self.player)
        self.player.display_inventory()
        self.board = Board(self.size)
        self.board.place(self.player.location, symbol="^")
        for m in self.monsters:
            self.board.place(m.location, "M")
        self.board.render()

    def fight(self, monster):
        while monster.health > 0 and self.player.health > 0:
            self.player.attack(monster)
            monster.attack(self.player)
            print(self.player)
            print(monster)
        print("FIGHT OVER")

    def get_choice(self, option_list):
        valid = False

        while not valid:

            for key in option_list:
                value = option_list[key]
                print(key, value)

            text = input("WHAT NOW?")

            if text in option_list:
                print(f"You chose option {text}")
                valid = True
            else:
                print(f"Invalid option {text}")
        return option_list[text]

    def add_random_item(self, character):
        data = [FirstAidKit(), FirstAidKit(), FirstAidKit(), Gold(100), Gold(500), Sword()]
        index = randint(0, len(data) - 1)
        character.collect(data[index])


game = Game()
# game.fight(Ogre())
# item = Gold(100)
# game.player.collect(item)
# game.player.collect(Sword())
# game.player.collect(FirstAidKit())
#
# game.player.display_inventory()
# print(game.player)

game.play()


