# import numpy as np
import matplotlib.pyplot as plt
from IPython import display

import sympy as sy
from sympy.abc import x, y  # folosim x si y ca simboluri matematice


# (1) calculam un "bounding box", care sa includa toate punctele de intersectie:
# pentru a-l calcula este nevoie sa gasim toate punctele de intersectie:
def intersection_points(equations):
    
    #functie pentru a intersecta 2 drepte:
    def intersect(d1, d2):
        # d se afla rezolvand sistemul
        a1, b1, c1 = d1
        a2, b2, c2 = d2
        solution = sy.solve([a1 * x + b1 * y + c1, a2 * x + b2 * y + c2], dict=True)[0]
        if not solution[x] or not solution[y]:
            return None
        else:
            return solution[x], solution[y]

    points = []
    for d1 in equations:
        for d2 in equations:
            intersection = None
            if d1 != d2:
                intersection = intersect(d1, d2)
            if intersection is not None:
                points.append(intersection)
    return points
def bounding_box(equations):
    intersections = intersection_points(equations)
    first_x, first_y = intersections[0]
    xmin, xmax = first_x, first_x
    ymin, ymax = first_y, first_y
    for x, y in intersections[1:]:
        if x < xmin:
            xmin = x
        elif x > xmax:
            xmax = x;
        if y < ymin:
            ymin = y
        elif y > ymax:
            ymax = y
    return (xmin, xmax), (ymin, ymax)


# pentru a realiza desenul de la (2)
# Aplicatiai ecuatiei in punctul x:
def calc_y(equation, x):
    a, b, c = equation
    # if ax + by + c == 0, then y is (-c - ax) / b
    y = (-c - a*x) / b
    return y
# Cream un semiplan pentru reprezentarea programului:
def create_semiplane(equation, xlim, ylim):
    a, b, c = equation
    if a == b and b == 0:
        raise ValueError(f'Constrangerea {equation} nu are coeficienti pentru x si y!')
    if a == 0:  # by + c == 0, deci y = -c / b
        # folosim xlim care este egal cu [xmin, xmax]
        direction = 'up' if b > 0 else 'down'
        return (xlim, [-c / b, -c / b], direction)
    if b == 0:  # ax + c == 0, deci x = -c / a
        # folosim ylim care este egal cu [ymin, ymax]
        direction = 'right' if a > 0 else 'left'
        return ([-c / a, -c / a], ylim, direction)
    # a si b sunt ambele nenegative => calculam ordonatele pt xmin, xmax
    direction = 'up' if b > 0 else 'down'
    return (xlim, [calc_y(equation, xlim[0]), calc_y(equation, xlim[1])], direction)

# desenam semiplane hasurate:
from matplotlib.patches import Polygon
def poligon_de_hasurat(inequation, color, hatch):
    # inequation is ([xmin, xmax], [applied_xmin, applied_xmax], direction)
    P1, P2, direction = inequation
    P1, P2 = (P1[0], P2[0]), (P1[1], P2[1])
    # hasuram din dreapta (xmin, applied_xmin)-() in directia direction:
    puncte = [P1, P2]
    if direction == 'right':
        puncte.append([P2[0] + 20, P2[1]])
        puncte.append([P1[0] + 20, P1[1]])
    elif direction == 'left':
        puncte.append([P2[0] - 20, P2[1]])
        puncte.append([P1[0] - 20, P1[1]])
    elif direction == 'up':
        puncte.append([P2[0], P2[1] + 20])
        puncte.append([P1[0], P1[1] + 20])
    elif direction == 'down':
        puncte.append([P2[0], P2[1] - 20])
        puncte.append([P1[0], P1[1] - 20])
    return Polygon(puncte, closed=True, fill=False, color=color, hatch=hatch)
def draw_semiplane(ax, semiplane, color, hatch):
    P0 = semiplane[0]  # x0, x1
    P1 = semiplane[1]  # y0, y1
    ax.plot(P0, P1, linewidth=3, color=color)
    ax.add_patch(poligon_de_hasurat(semiplane, color, hatch=hatch))

    
# (1) cream figura si calculam un bounding box pentru program

# cream figura:
fig, ax = plt.subplots(1, 1)

# introducem datele:
# Sample objective Function: Minimise 225x + 200y
c = (225, 200)
# Constraints: y ≥ 25; x ≥ 40; x + y ≤ 150 (*)
constraints = [
    (0, 1, -25),  # triplet (a,b,c) cu a*x + b*y + c >= 0
    (1, 0, -40),
    (-1, -1, 150)]

# Gasim un bounding box care contine toate intersectiile de puncte dintre drepte:
bound = bounding_box(constraints)
xlim, ylim = bound

# (2) Cream un desen pentru a vizualiza toate semiplanele:
# desenam dreptele:
semiplanes = [create_semiplane(c, xlim, ylim) for c in constraints]
# print(xlim, ylim)
# print(semiplanes)
# adaugam o culoare si un tip de hasurare fiecarui semiplan:
colors, hatches = {}, {}
all_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
all_hatches = ['-', '+', 'x', '\\', '*', 'o', 'O', '.', '/', '|']
for i, semiplane in enumerate(semiplanes):
    colors[i] = all_colors[i % len(all_colors)]
    hatches[i] = all_hatches[i % len(all_hatches)]
    
for i, semiplane in enumerate(semiplanes):
    draw_semiplane(ax, semiplane, colors[i], hatches[i])

intersectii = intersection_points(constraints)


for i in intersectii:
    plt.scatter(i[0], i[1], marker='o', color='black', linewidth=.1, zorder=5)

print('In aceasta celula de cod am exemplificat cum se suprapun semiplanurile si cum am ales un bounding box pentru reprezentarea grafica, indiferent de datele problemei.')
print(f'Am obtinut urmatoarele planuri, care contin [x1, x2], [y1, y2] si directia planului: {semiplanes}')
plt.show()
# dupa rularea acestei celule, putem observa un exemplu de program liniar bounded,
# adica am obtinut o regiune fezabila (care respecta conditiile necesare) finita, cuprinsa intre cele trei linii colorate si determinata de cele 3 puncte.
#   Deoarece cele 3 semiplanuri determina o suprafata convexa finita, solutia este unul dintre varfurile acesteia
#   Celelalte cazuri posibile sunt:
#    * sa obtinem un fascicul de drepte, iar intersectia cu Ox sau Oy a uneia dintre drepte sa fie solutia.
#     Acest lucru se intampla deoarece in toate programele liniare avem x și y >= 0
#    * sa obtinem o suprafata infinita spre Ox, Oy, sau ambele. In acest caz, se poate ca maximul sau miniml sa fie nedeterminate (plus sau minus infinit),
#     sau sa se gaseasca la intersectia a doua semiplanuri care marginesc inferior suprafata fezabila.

# Pentru a rezolva complet un program, este nevoie sa calculam suprafata fezabila si sa gasim varful (daca este finita)
# sau dreapta care intersecteaza Ox sau Oy
class FeasibleRegion(object):
    
    bound = None
    xlim, ylim = None, None
    
    def __init__(self, objective, equations):
        '''
        ecuatiile sunt de tipul (a,b,c) cu ax+by+c>=0.
        In __init__ initializam obiectul prin a transforma ecuatiile intr-un bounding box, si ecuatii de semiplane ca in celula anterioara de cod
        '''
        self.objective = objective
        self.equations = equations.copy()
        
        # calculam bounding box-ul, 
        self.bound = bounding_box(constraints)
        self.xlim, self.ylim = bound
        
        # calculam semiplanurile
        self.semiplanes = []
        for eq in equations:
            self.semiplanes.append(create_semiplane(eq, xlim, ylim))
        
    
    #functie care intersecteaza doua dintre ecuatiile stocate in obiect
    def intersect_equations(self, index1, index2):
        # creaza o lista de perechi de indici, care marcheaza care ecuatii (si semiplane, 
        #   deoarece self.equations[i] corespunde lui self.semiplanes[i]) au puncte comune
        a1, b1, c1 = self.equations[index1]
        a2, b2, c2 = self.equations[index2]
        solution = sy.solve([a1 * x + b1 * y + c1, a2 * x + b2 * y + c2], dict=True)[0]
        if not solution[x] or not solution[y]:
            return None
        else:
            return solution[x], solution[y]
    def get_all_intersections(self):
        intersections = []
        for i1 in range(len(self.equations)):
            for i2 in range(i1 + 1, len(self.equations)):
                intersection = self.intersect_equations(i1, i2)
                if intersection:
                    intersections.append((i1, i2, intersection))
        return intersections
    def is_point_feasible_for_equation(self, point, index):
        x, y = point
        a, b, c = self.equations[index]
        # cazurile sunt:
        applied = a * x + b * y + c;
        # este pe dreapta semiplanului:
        #   sau este in interiorul regiunei fezabile a semiplanului:
        eps = .001
        if abs(applied) >= eps:
            return True
        else:
            return False
    
    def _get_looser_semiplane(self, i1, i2):
        # equation[i1] and equations[i2] must denote parallel lines. We keep the looser constraint!
        a1, b1, c1 = self.equations[i1]
        a2, b2, c2 = self.equations[i2]
        if a1 != a2 or b1 != b2:
            return None
        dir1 = self.semiplanes[i1][-1]
        dir2 = self.semiplanes[i2][-1]
        if dir1 != dir2:
            return None
        if a1 == 0 and b1 == 0:
            raise ValueError(f"Planes {self.semiplanes[i1]}, {self.semiplanes[i2]}has both x and y coeficients equal to zero")
        if a1 == 0: # horizontal lines
            # b1 * y >= -c1  and  b2 * y >= -c2
            return i1 if -c1 / b1 > -c2 / b2 else i2
        if b1 == 0:
            # a1 * x >= -c1  and  a2 * x >= -c2
            return i1 if -c1 / a1 > -c2 / a2 else i2
        # both a1 and b1 are non-zero
        x0 = self.xlim[0]
        y0 = (-a1 * x0 - c1) / b1   # a1 * x + b1 * y + c1 == 0   means that   y=(-a1 * x - c1) / b1
        applied = a2 * x0 + b2 * y0 + c2
        return i1 if applied >= 0 else i2
            
       
    def compute_region(self):
        # remove parallel semiplanes, and also the equations to keep index-correlation TODO use dicts so you don't modify in two places!
        to_delete = []
        for i1, plane1 in enumerate(self.semiplanes):
            for i2, plane2 in enumerate(self.semiplanes):
                if i1 == i2:
                    continue
                looser_semiplane = self._get_looser_semiplane(i1, i2)
                if looser_semiplane:    
                    to_delete.append(looser_semiplane)
        for i in to_delete:
            del self.equations[i]
            del self.semiplanes[i]
        # pentru a calcula regiunea, calculam toate punctele de intersectie (perechi de indici i1, i2)
        intersections = self.get_all_intersections()
        # remove extra points
        index = 0
        while index < len(intersections):
            i, j, point = intersections[index]
            # este punctul point in fezabil? (conform equations[i] și equations[j], este. Urmeaza testate si celelalte ecuatii)
            is_point_feasible = True
            for k in range(len(self.semiplanes)):
                if i == k or j == k:
                    continue
                if not self.is_point_feasible_for_equation(point, k):
                    is_point_feasible = False
            print(point, is_point_feasible)
            if not is_point_feasible:
                # remove the point
                del intersections[index]
            else:
                index += 1
        print(intersections)
        # TODO coalesce the remaining points into 1 component
        rect = []
        adjacent = {}
        for i, j, _ in intersections:
            if not adjacent.get(i):
                adjacent[i] = [j]
            else:
                adjacent[i].append(j)
        print(adjacent)
        # start with first index:
        i = min(list(adjacent.keys()))
        queue = [i]
        rectangle = [(i, None, None)]
        print(f'Queue is {queue}')
        while queue:
            # for each node, find the intersecting points
            curr = queue.pop()
            print(f'Popped {curr}. Queue becomes {queue}.')
            # TODO when getting the adjacent planes, start creating the rect
            for i in adjacent.get(i, []):
                queue.append(i)
            print(f'Now queue is {queue}')
            
            

# introducem datele:
# Sample objective Function: Minimise 225x + 200y
c = (225, 200)
# Constraints: y ≥ 25; x ≥ 40; x + y ≤ 150 (*)
constraints = [
    (0, 1, -25),  # triplet (a,b,c) cu a*x + b*y + c >= 0
    (1, 0, -40),
    (-1, -1, 150)]
region = FeasibleRegion(c, constraints)
region.compute_region()

import re

# Vom rezolva problema folosind aceiasi metoda ca de la examen

# (1) Aducem la forma standard
next_s_index = 1
def parse_constraint(string):
    # returneaza un triplet (a,b,c) pentru forma standard (ax+by+c>=0)
    # variables are of type x1, x2, x3 ..
    triples = re.findall(r'(-|\+)?\s*(\d*)(\s*\*\s*)?(x\d+)', string)
#     print(triples)
    coefs = { triple[3]: (-1 if triple[0] == '-' else 1) * (1 if triple[1] == '' else int(triple[1])) for triple in triples }
    sign = re.search('(>=)|(<=)|(==)|(=)', string)
    if not sign:
        return None
    else:
        rhs = re.search('\d+', string[sign.end():])
        sign = sign.group(0)
    if not rhs:
        return None
    rhs = int(rhs.group(0))
    # negam inegalitatea, daca partea dreapta a constrangerii este negativa
    if rhs >= 0:
        coefs['rhs'] = rhs
    else:  # negam:
        coefs = {var: -coef for var, coef in coefs.items()}
        coefs['rhs'] = -rhs;
    # scapam de inegalitate:
    global next_s_index
    if sign == '>=':  # '=' or '=='
        coefs[f's{next_s_index}'] = 1
        next_s_index += 1
    elif sign == '<=':
        coefs[f's{next_s_index}'] = -1
        next_s_index += 1
    return coefs
def parse_objective(string):
    triples = re.findall(r'(-|\+)?\s*(\d*)(\s*\*\s*)?(x\d+)', string)
    coefs = {
        triple[3]: (-1 if triple[0] == '-' else 1) * (1 if triple[1] == '' else int(triple[1])) for triple in triples}
    return coefs


# algoritmul simplex pentru minimizarea unei functii:
def minimize(constraints_strings, objective_string):
    constraints = [parse_constraint(c) for c in constraints_strings]
    objective = parse_objective(objective_string)

    # print(constraints)
    # print()
    # print(objective)

    #gasim sau adaugam variabilele pentru o baza:
    variables = set(list(objective.keys()))
    objective_variables = list(objective.keys())
    basic_variables = {}
    for i1, c1 in enumerate(constraints):
        # verificam daca avem o variabila de baza in constrangere
        for var in c1.keys():
            is_basic = True
            for i2, c2 in enumerate(constraints):
                if i1 != i2 and c2.get(var):
                    is_basic = False
                    break
            if is_basic and c1[var] == 1:
                basic_variables[i1] = var
        # adaugam variabilele la lista tuturor variabilelor
        variables.update(c1.keys())
    variables -= {'rhs'}
    
    for i in range(len(constraints)):
        if not basic_variables.get(i):  # nu avem variabila de baza pt constrangere:
            global next_s_index
            basic_variables[i] = f's{next_s_index}'
            constraints[i][f's{next_s_index}'] = 1  # adaugam variabila artificiala
            objective[f's{next_s_index}'] = '-M'  # deoarece minimizam
            next_s_index += 1

    print('Am ales variabilele pentru baza:')
    for i, constraint in enumerate(constraints):
        print('Pentru constrangerea', constraint)
        print('\tam ales variabila de baza', basic_variables[i])
    print()
    print(f'Functia obiectiv devine {objective}')
    
    # adaugam constrangerea -objective + new_slack_variable = 0
    # construim tabelul simplex:
    
    print()
    print(f'Variabilele initiale are functiei obiectiv sunt {objective_variables}; Variabilele de baza sunt {list(basic_variables.values())}; Restul variabilelor sunt {variables - set(objective_variables) - set(basic_variables.values())}')
    

# exemplu cu initializari in loc de citire de la tastatura:
last_s_index = 1
constraints = [
    ' x1 - 2*x2 + x3 >= 0',
    '-x1        + x3 <= 3',
    '-x1        + x3 == 5',
]
objective = '22*x1 + x2 + 3*x3'
minimize(constraints, objective)
# !!! Ruleaza tema 3, dar cu citirea de la tastatura.
# !!! Este nevoie rularea celulei precedente !!!

print('''HELP:

Introduceti constrangeri de urmatoarele tipuri:
    
     x1 - 2*x2 + x3 >= 0 (inegalitate <=)
    -x1        + x3 <= 3 (inegalitate >=)
           -x2 + x3 = -2 (egalitate)

Apoi introduceti functia obiectiv, sub acceasi forma (cu '*' pentru orice inmultire)
''')
num = int(input('Numarul constrangerilor: '))
c = []
for i in range(num):
    constraint = parse_constraint(input(f'Constrangerea {i + 1}: '))
    print(constraint)
    c.append(constraint)
print(c)
z = parse_objective(input("Introduceti functia obiectiv:"))