name = "jan kowalski"
print(name.title())
print(name.lower())
print(name.upper())
def full_name():
    first_name = input("Provide your first name: ")
    last_name = input("Provide your second name: ")
    full_name = first_name + " " + last_name
    message = "Witaj, " + full_name.title() + "!"
    return(message)
full_name()
import this
bicycles = ["trekingowy", "górski", "miejski", "szosowy"]
print(bicycles[2].title())
print("Ostatni element: " + bicycles[-1].title())
#zmiana

moto = ['honda','yamaha','suzuki']
print(moto)
moto[0] = 'zmieniona nazwa'
print(moto)
#dodanie
moto.append('dodana wartość')
print(moto)
dynamiczna_lista = []
def dodaj_element():
    element_listy = input('Dodaj do listy: ')
    dynamiczna_lista.append(element_listy)
    quest = input("Chcesz jeszcze coś dodać coś do listy? y/n")

    if quest == "n":
        print(dynamiczna_lista)
    else:
        dodaj_element()

dodaj_element()
nowa_lista = []
nowa_lista.append('ładna wartość')
nowa_lista.append('bardzo ładna wartość')
nowa_lista.append('brzydka wartość')
print(nowa_lista)

print(nowa_lista)
del nowa_lista[-1]
print(nowa_lista)
motorcycles = []
motorcycles.append('honda')
motorcycles.append('yamaha')
motorcycles.append('suzuki')

# pokazuje element usunięty
motorcycles_popped = motorcycles.pop()
print(motorcycles)
print(motorcycles_popped)

first_owned = motorcycles.pop(0)
print('Mój pierwszy motor to ' + first_owned +'.')
motorcycles = []
motorcycles.append('c')
motorcycles.append('c2')
motorcycles.append('c3')

print('Removed by value - before: '+ str(motorcycles))

motorcycles.remove('c3')
print('Removed by value - after: '+ str(motorcycles))
cars = ['bmw', 'audi', 'toyota', 'subaru']
cars.sort()

cars_reverse = ['bmw', 'audi', 'toyota', 'subaru']
cars_reverse.sort(reverse=True)
print(cars)
print(cars_reverse)
cars = ['bmw', 'audi', 'toyota', 'subaru']
print("lista początkowa: " + str(cars))

cars_sorted = sorted(cars)
print("lista tymczasowo posortowana: " + str(cars_sorted))
print("Znowu lista wyjściowa: " + str(cars))
cars = ['bmw', 'audi', 'toyota', 'subaru']
len(cars)
magicians = ['dumbledor', 'snape', 'potter']
for magician in magicians:
    print(magician.title() + " - jest wspaniałym czarodziejem!")
for value in range(1,5):
    print(value)
# range w tworzeniu listy
numbers = list(range(1,10))
print(numbers)
#even numbers from list 
even_numbers = list(range(0,11,2))
print(even_numbers)
digits = [1,2,3,4,5,6,7,8,9,10]
digits_min = min(digits)
digits_max = max(digits)
digits_sum = sum(digits)
print("Min: " + str(digits_min) + "\n" +"Max: " + str(digits_max) + "\n" + "Suma: " + str(digits_sum))
squares = [value**2 for value in range(2,5)]
print(squares)
players = ['ana', 'kristina', 'stas', 'rafał', 'mati']
print(players[0:2])

## zwracanie trzech ostatnich elementów listy
print(players[-3:])
print("Oto dwóch pierwszych graczy na naszej liście: ")
for player in players[0:2]:
    print(player.title())
my_foods = ['pizza','falafel','ciasto z marchewki']
friend_foods = my_foods[:]
print("Moje ulubione jedzenie: " + str(my_foods) + "\n" + "Ulubione jedzenie mojego ziomka: " + str(sorted(friend_foods)))
values = (200, 50)
print(values)

#wywołujemy błąd żeby pokazać, że krotka jest niemodyfikowalna! 'tuple' object does not support item assignment
values[0] = (250)
print(values)
for value in values:
    print(value)
cars = ['audi', 'bmw', 'subaru', 'toyota']

for car in cars:
    if car == 'bmw':
        print(car.upper())
    else:
        print(car.title())
car = 'bmw'
car == 'bmw'
car = 'BMW'
car == 'bmw'
car = 'BMW'
car.lower() == 'bmw'
requested_toppings = ['pieczarki', 'cebula', 'kukurydza']
'pieczarki' in requested_toppings
'ananas' in requested_toppings
ingredients = ['pepperoni', 'salami', 'corn']
my_pizza = []
print("Dostępne składniki: \n" + str(ingredients))
def dodaj_skladnik():
    ingredient = input('Jaki składnik chcesz dodać do pizzy?')
    if ingredient in ingredients:
        my_pizza.append(ingredient)
    else:
        quest = input("Nie mamy " + ingredient + " :( Chcesz dodać składnik do listy możliwych dodatków? t/n")
        if quest == 't' or quest == 'T':
            ingredients.append(ingredient)
            my_pizza.append(ingredient)
            print('Możliwe składniki: ' + str(ingredients))
            print('Twoja pizza: ' + str(my_pizza))

    want_more = input("Chcesz dodać jeszcze coś do swojej pizzy? t/n")
    if want_more == 't':
        dodaj_skladnik()
    else:
        print("Twoje dodatki to: " + str(my_pizza))
dodaj_skladnik()
banned_users = ['andrzej', 'karolina', 'rafał']
user = 'maria'

if user not in banned_users:
    print(user.title() + ', dodaj komentarz :)')
#problem praktyczny: cena biletu do kina zależna od wieku
print("Kino zaprasza!")
def ticket_price():
    age = int(input("Podaj swój wiek: "))
    if age <= 13:
        price = 13
    elif age > 13 and age <=18:
        price = 15
    else:
        price = 26
    
    print("Koszt biletu wynosi: " + str(price) +"zł")
ticket_price()
# sprawdzanie czy lista nie jest pusta

# requested_elements = [1,2,3,4,5,6]
requested_elements = []
if requested_elements:
    for requested_element in requested_elements:
        print("e:" + str(requested_element))
else:
    print("Ziomuś! Twoja lista jest pusta :( ")
alien_0 = {'color': 'green', 'points': 5}
print(alien_0['color'])
print(alien_0['points'])

## dodanie nowych wartości do słownika
alien_0['x_position'] = 0
alien_0['y_position'] = 25
print(alien_0)
#nadpisywanie wartości słownika
alien_0['color'] = 'yellow'
print("Teraz obcy ma kolor " + str(alien_0['color']))
alien_1 = {'position_x':0, 'position_y':25, 'speed':'medium'}
print("Początkowa wartość position_x to: " + str(alien_1['position_x']))
#przesunięcie obcego w prawo
# ustalenie odległości, jaką powinien pokonać obcy poruszający się z daną szybkością

if alien_1['speed'] == 'slow':
    x_increment = 1
elif alien_1['speed'] == 'medium':
    x_increment = 2
else:
    x_increment = 3

#nowe położenie to suma dotychczasowego położenia i wartości x_increment
alien_1['position_x'] = alien_1['position_x'] + x_increment

print("Nowa wartość position_x to: " + str(alien_1['position_x']))
print(alien_0)
del alien_0['points']
print(alien_0)
#Ankieta dotycząca ulubionego języka programowania
favorite_languages = {
    'user1': 'python',
    'user2': 'c',
    'user3': 'ruby',
    'user4': 'python', #dobrą praktyką jest zostawianie tu przecinka, żeby słownik był gotowy na kolejną pare
}

print("Ulubiony język użytkownika 1 to: " + favorite_languages['user1'].title() + " :)")
user_0 = {
    'username': 'jkowalski',
    'first_name': 'jan',
    'last_name': 'kowalski',
}
print(user_0)

for key, value in user_0.items():
    print("\nKlucz: " + key)
    print("Wartość: " + value)
print(favorite_languages)

for user_id, language in favorite_languages.items():
    print("Ulubiony język programowania " + user_id + " to " + language + '.\n')
for name in favorite_languages.keys():
    print(name)
for name in favorite_languages:
    print(name)
friends = ['user2', 'user4']

for name in favorite_languages.keys():
    print(name.title())
    
    if name in friends:
        print(" Witaj, " + name.title() + " widzę, że Twoim ulubionym językiem jest " + favorite_languages[name])
for name in sorted(favorite_languages.keys()):
    print(name.title() + ", dziękujemy bardzo za udział w ankiecie ;)")
print("W ankiecie zostały wymienione następujące języki programowania: ")
for value in favorite_languages.values():
    print(value.title())
print("W ankiecie zostały wymienione następujące języki programowania: ")
for value in set(favorite_languages.values()):
    print(value.title())
alien_0 = {'color':'green', 'points':0}
alien_1 = {'color':'yellow', 'points':10}
alien_2 = {'color':'red', 'points':15}

aliens = [alien_0, alien_1, alien_2]
print(aliens)
aliens = []

#utworzenie 27 obcych o kolorze zielonym
for alien_number in range(27):
    new_alien = {'color': 'green', 'points': 5, "speed": 'slow'}
    aliens.append(new_alien)

#wyświetlenie pierwszych 5 obcych
for alien in aliens[:5]:
    print(alien)

print("Calkowita liczba wrzuconych obcych to: " + str(len(aliens)))
#zmiana koloru obcego

for alien in aliens[0:3]:
    if alien['color'] == 'green':
        alien['color'] = 'blue'
        alien['points'] = 10
        alien['speed'] = 'medium'

for alien in aliens[:10]:
    print(alien)
print(favorite_languages)
favorite_languages['user1'] = ['python', 'c++', 'js']
favorite_languages['user2'] = ['ruby']
favorite_languages['user3'] = ['c', 'c++']
favorite_languages['user4'] = ['python']
print(favorite_languages)
for name, languages in favorite_languages.items():
    print("\n Favorite languages for: " + name + " is:")
    for language in languages:
        print('\t' + language.title())
users = {
    'aeinstein': {
        'first_name': 'albert',
        'last_name': 'einstein',
        'location': 'princeton',
    },
}

def add_user():
    nick = input("Wprowadź swój nickname: \n")
    if nick not in users:
        print("Igła")
        first_name = input("Twoje imie: \n")
        last_name = input("Twoje nazwisko: \n")
        location = input("Wprowadź swoją lokalizację: \n")
        users[nick] = {'first_name': first_name, 'last_name':last_name, 'location':location}
    else:
        print("Niestety wybrana nazwa użytkownika jest zajęta :( Spróbuj ponownie! ")
        add_user()
add_user()
print('Lista użytkowników: ')
for user, user_info in users.items():
    print(user)
    
    print("\t Imię: "+ user_info['first_name'].title() + 
          "\n\t Nazwisko: " + user_info['last_name'].title()+ 
          "\n\t Lokalizacja: " + user_info['location'].title())
# print(users)
prompt = "\n Powiedz mi cokolwiek o sobie a wyświetlę to na ekranie."
prompt += "Napisz koniec żeby zakończyć działanie programu."

message=""

while message != 'koniec':
    message = input(prompt)
    
    if message != 'koniec':
        print(message)
active = True
while active:
    message = input("Napisz coś. Dodaj 'koniec' jeśli chcesz zakończyć działanie programu")
    
    if message == 'koniec':
        active = False
    else:
        print(message)
pizza_toppings = []
available_toppings = ['corn', 'salami', 'pepperoni']
quest = "Dodaj kolejny składnik do pizzy. W przeciwnym wypadku zakończ zamówienie poprzez komende 'koniec' :)"

active = True
while active:
    message = input(quest)
    
    def add_topping():
        topping = message
        if topping in available_toppings:
            pizza_toppings.append(topping)
        else:
            print("Niestety nie mamy takiego czegoś. Spróbuj czegoś innego, albo zakończ program poprzez 'koniec' ")

    if message == 'koniec':
        active = False
        print('Twoja pizza składa się z: ')
        for pizza_topping in pizza_toppings:
            print('\t - ' + pizza_topping)
    else:
        add_topping()
prompt = "\n Podaj nazwy miast, które chciałbyś odwiedzić"
prompt += "\n Gdy zakończysz podawanie miast, napisz 'koniec'"

while True:
    city = input(prompt)
    
    if city == 'koniec':
        break
    else:
        print("Chciałabym odwiedzić " + city.title() + '!')
#rozpoczynamy od użytkowników, którzy mają być zweryfikowani.
#tworzymy pustą listę przeznaczoną do przechowywania zweryfikowanych użytkowników
unconfirmed_users = ['ala', 'bartek', 'kasia']
confirmed_users = []
to_verify = []
#weryfikujemy poszczególnych userów, póki lista nie będzie pusta

while unconfirmed_users:
    current_user = unconfirmed_users.pop()
    print("Weryfikacja użytkownika: " + current_user.title() + "...")
    
    quest = input("Czy dane użytkownika są poprawne? t/n")
    
    if quest == 't':
        confirmed_users.append(current_user)
    else:
        print("Dodaję do listy użytkowników nadających się do ponownej weryfikacji")
        to_verify.append(current_user)

print("Zweryfikowano wymienionych poniżej użytkowników: ")
for confirmed_user in confirmed_users:
    print(confirmed_user.title())
pets = ['kot', 'pies', 'ryba', 'kot', 'filemon', 'złota rybka', 'łosoś']

while 'kot' in pets:
    pets.remove('kot')

print(pets)
responses = {}

#ustawienie flagi wskazującej, czy ankieta jest aktywna
polling_active = True

while polling_active:
    name = input("Jak masz na imie? ----- \t ")
    response = input("Co lubisz robić najbardziej? ----- \t ")
    
    responses[name] = response
    
    repeat = input("Chcesz wypełnić ankietę? t/n ----- \t")
    if repeat != 't':
        polling_active = False
    
print("\n ----Wyniki ankiety----")
for name, response in responses.items():
    print(name + " najbardziej lubi: " + response + " :)")
def greet_user():
    """ Wyświetla proste powitanie. """
    print("No cześć!")
    
greet_user()
def greet_current_user(username):
    """ Wyświetla powitanie dla current user """
    print("Hi, " + username.title() + '!')
    
greet_current_user("Janusz")
def describe_pets(animal_type, pet_name):
    """Wyświetla informacje o zwierzęciu."""
    print("Moje zwierzę to: " + animal_type)
    print("Mój "+ animal_type + " ma na imię " + pet_name.title())

describe_pets("chomik","janusz")
describe_pets("kot","filemon")

