class Mother:
    
    def __init__(self, attr1, attr2):
        self.attr1 = attr1
        self.attr2 = attr2
        
    def method1(self):
        print(f'method1: {self.attr1}')
    
    def method2(self, attr3):
        print(f'method2: {self.attr2}, {attr3}')

class Child(Mother):
    pass

m = Mother(1, 2)
m.method1()
m.method2(3)
c = Child(4, 5)
c.method1()
c.method2(6)
class Party:
    
    def __init__(self, name, age_limit):
        self.name = name
        self.age_limit = age_limit
        self.guests = []
        
    def __str__(self):
        return f'Lista de convidados:'
    
    def invite(self, guest):
        if type(guest) != Guest:
            print('Entre com um objeto do tipo Guest')
        elif guest.age < self.age_limit:
            print(f'O convidado {guest.name} é muito jovem para esta festa')
            print(f'A idade mínima é {self.age_limit}, o convidado possui apenas {guest.age} anos')
        else:
            self.guests.append(guest)
            guest.invite(self)
            print('A lista de convidados foi atualizada')
            print(str(self))
class Guest:
    
    def __init__(self, name, age, phone):
        self.name = name
        self.age = age
        self.phone = phone
        self.party = None
        
    def __repr__(self):
        return f'({self.name}, {self.age}, {self.phone})'
    
    def invite(self, party):
        self.party = party
        print(f'{self.name} foi convidado para a festa {party.name}')
    
    def greetings(self):
        print('Ola, sou um dos convidados da festa')
        print(f'Meu nome é {self.name}')
        print(f'Minha idade é {self.age}')
        
    def call_to_phone(self):
        print(f'Ligando para {self.phone}...')
        print(f'Fim da ligação')
class VipGuest(Guest):
    
    def __init__(self, name, age, phone):
        super().__init__(name, age, phone)
        
    def get_free_drink(self):
        print(f'Aqui está seu drink, {self.name}')
matheus = VipGuest('Matheus', 24, 988551234)
alfredo = Guest('Alfredo', 23, 988226789)
allan = Guest('Allan', 18, 988772456)

confra_ia = Party('Confra IA 2020', 21)
confra_ia.invite(matheus)
confra_ia.invite(alfredo)
confra_ia.invite(allan)

matheus.get_free_drink()
class Party:
    
    def __init__(self, name, age_limit):
        self.name = name
        self.age_limit = age_limit
        self.guests = []
        
    def __str__(self):
        return f'Lista de convidados: {self.guests}'
    
    def invite(self, guest, is_vip=False):
        if type(guest) != Guest:
            print('Entre com um objeto do tipo Guest')
        elif guest.age < self.age_limit:
            print(f'O convidado {guest.name} é muito jovem para esta festa')
            print(f'A idade mínima é {self.age_limit}, o convidado possui apenas {guest.age} anos')
        else:
            self.guests.append(guest)
            guest.invite(Invite(guest.name, self.name) if is_vip else VipInvite(guest.name, self.name))
            print('A lista de convidados foi atualizada')
            print(str(self))
    
    def get_free_drink(self, guest):
        if type(guest.show_invite()) == VipInvite:
            print(f'Aqui está seu drink, {guest.name}')
        else:
            print('Você não tem direito a drinks grátis')
class Invite:
    
    def __init__(self, guest_name, party_name):
        self.guest_name = guest_name
        self.party_name = party_name
        
    def read(self):
        print(f'{self.guest_name} foi convidados para a festa {self.party_name}')
        
    def __repr__(self):
        return 'Normal'
class VipInvite:
    
    def __init__(self, guest_name, party_name):
        self.guest_name = guest_name
        self.party_name = party_name
        
    def read(self):
        print(f'{self.guest_name} foi convidados para a festa {self.party_name}, como convidado VIP')
        
    def __repr__(self):
        return 'VIP'
class Guest:
    
    def __init__(self, name, age, phone):
        self.name = name
        self.age = age
        self.phone = phone
        self.invite_card = None
        
    def __repr__(self):
        return f'({self.invite_card} | {self.name}, {self.age}, {self.phone})'
    
    def invite(self, invite):
        self.invite_card = invite
        
    def show_invite(self):
        return self.invite_card
    
    def greetings(self):
        print('Ola, sou um dos convidados da festa')
        print(f'Meu nome é {self.name}')
        print(f'Minha idade é {self.age}')
    
    def call_to_phone(self):
        print(f'Ligando para {self.phone}...')
        print(f'Fim da ligação')
matheus = Guest('Matheus', 24, 988551234)
alfredo = Guest('Alfredo', 23, 988226789)
allan = Guest('Allan', 18, 988772456)

confra_ia = Party('Confra IA 2020', 21)
confra_ia.invite(matheus)
confra_ia.invite(alfredo, is_vip=True)
confra_ia.invite(allan)

confra_ia.get_free_drink(matheus)
confra_ia.get_free_drink(alfredo)
confra_ia.get_free_drink(allan)
import datetime
from time import sleep

sleep(1)
datetime.datetime(year=2020, month=9, day=17)
filename = 'data.txt'

file = open(filename, 'w')
file.close()

file = open(filename, 'r')
file.close()

file = open(filename, 'a')
file.close()
filename = 'data.txt'

file = open(filename, 'w+')
file.write('Olá Mundo')
file.seek(0)
print('==========')
print(file.read())
file.close()

file = open(filename, 'wb+')
file.write(b'Ola Mundo')
file.seek(0)
print('==========')
print(file.read())
file.close()

file = open(filename, 'a+')
file.write('\nOutra linha')
file.seek(0)
print('==========')
print(file.readline())
print(file.readline())
print('==========')
file.close()
filename = 'data.txt'

with open(filename) as file:
    line = file.readline()
    print(line)
class BankAccount:
    
    def __init__(self, start_amount=0):
        self.amount = start_amount
        self.backup = 0
        
    def deposit(self, value):
        self.amount += value
    
    def withdraw(self, value):
        if self.amount - value < 0:
            raise Exception(f'Você não pode sacar {value}, porque só possui {self.amount}') # Usado para criar um erro
        self.amount -= value
            
    def __enter__(self):
        # Salva um backup do valor inicial para recuperar se houver erro
        self.backup = self.amount
        return self
    
    def __exit__(self, type_, value, traceback):
        # Se houve um erro, então retorna o valor de backup
        if isinstance(value, Exception):
            print(f'Restaurando backup. Valor inicial: {self.backup}')
            self.amount = self.backup

try:
    with BankAccount(100) as acc:
        print(acc.amount)
        acc.deposit(50)
        print(acc.amount)
        acc.withdraw(75)
        print(acc.amount)
        acc.deposit(50)
        print(acc.amount)
        acc.withdraw(150) # Vai dar erro, pois so tem 125 na conta
except:
    pass
finally:
    print(acc.amount) # Neste ponto, não obtemos 125, mas 100, que foi o valor inicial
import re

text = 'avc abc abc adc'
print(re.findall(r'abc', text))
print(re.search(r'abc', text))
pattern = re.compile(r'abc')
for match in pattern.finditer(text):
    print(match)
print(pattern.match('abc'))
print(pattern.match('abc '))
print(pattern.fullmatch('abc'))
print(pattern.fullmatch('abc '))
text = 'banana, maçã, laranja, melancia'
print(re.split(r',', text))
print(re.split(r', ', text))
print(re.sub(r'banana', 'pessego', text))
import re
emails = [
    'matheus.santos@edge.ufal.br',
    'matheus.santos@edge.ufal.br.al',
    '@edge.ufal.br',
    'matheus#santos@edge.ufal.br',
    'matheus.dos.santos@edge.ufal.br'
]

patterns = [r'@edge.ufal.br$',                   # Encontra todo email que possua @edge.ufal.br no final
            r'^matheus',                         # Encontra todo email que possua matheus no inicio
            r'^.*@edge.ufal.br$',                # Encontra todo email que possua zero ou mais caracteres no inicio e @edge.ufal.br no final
            r'^.+@edge.ufal.br$',                # Encontra todo email que possua um ou mais caracteres no inicio e @edge.ufal.br no final
            r'^[a-zA-Z0-9.\-_]+@edge.ufal.br$',  # Encontra todo email que possua um ou mais: letras maiusculas, ou letras minusculas, ou digitos ou ponto, ou hífen ou undeline no inicio e @edge.ufal.br
            r'^[\w.\-]+@edge.ufal.br$',          # Encontra todo email que possua um ou mais: letras maiusculas, ou letras minusculas, ou digitos ou ponto, ou hífen ou undeline no inicio e @edge.ufal.br
            r'^[^#]+@edge.ufal.br$',             # Encontra todo email que possua um ou mais caracter diferente de #, no inicio e @edge.ufal.br no final
            r'^[^#\s]+@edge.ufal.br$',           # Encontra todo email que possua um ou mais caracter diferente de #, ou diferente de espaços vazios, no inicio e @edge.ufal.br no final
            r'^[^#\s]{14}@edge.ufal.br$']        # Encontra todo email que possua, exatamente, 14 caracteres diferente de #, ou diferente de espaços vazios, no inicio e @edge.ufal.br no final

for p in patterns:
    print(f'=====Pattern {p}=====')
    for e in emails:
        print(re.findall(p, e))
    print(f'=============' + ('=' * len(p)) + '=====')