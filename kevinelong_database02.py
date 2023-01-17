import sqlite3
import random


class MyDatabase:

    def __init__(self, database_name='test.db'):
        # connect to db get cursor
        self.connection = sqlite3.connect(database_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    # create table
    def create_table(self):
        self.cursor.executescript("DROP TABLE IF EXISTS Pets;")
        self.cursor.executescript("CREATE TABLE Pets(Id INT, Name TEXT, Price INT);")
        self.connection.commit()

    # insert record
    def create_pet(self, pet_name, pet_price=100):
        self.cursor.execute(f"INSERT INTO Pets VALUES({random.randint(1, 999999999999)}, '{pet_name}', {pet_price});")

    def get_pets(self):
        self.cursor.execute("SELECT * FROM Pets")
        return self.cursor.fetchall()


def display_pets(data):
    output = []

    for row in data:
        print(row)
        output.append(f"ID={row[0]}, NAME={row[1]}, PRICE={row[2]}.")

    print("<br>\n".join(output))


db = MyDatabase('test.db')

db.create_pet("Rabbit", 900)
db.create_pet("Dog", 1100)
db.create_pet("Cat", 1200)

data = db.get_pets()

display_pets(data)
db.connection.close()

