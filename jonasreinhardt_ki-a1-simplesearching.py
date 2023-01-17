# Die maximale Anzahl an Personen, die im Boot mitfahren können
max_boat = 2

# Wie viele Missionare und Kannibalen jeweils zu Beginn am Fluss stehen
m_start = 3
k_start = 3

# Der Zustand beschreibt immer die Situation auf der Startseite des Flusses.
# Die drei Werte der Liste stehen jeweils für die Menge der Kannibalen,
# Missionare und ob das Boot anliegt (1 wenn ja, 0 wenn nicht)
startzustand = [m_start, k_start, 1]

# Der Zielzustand ist üblicherweise die Situation, dass alle Personen den Fluss überquert haben
zielzustand = [0, 0, 0]
# Erzeuge alle denkbaren Folgezustände, wir schauen nicht auf feasibility
# Wir geben eine Liste von Listen (den Zuständen) zurück
def gib_folgezustaende(zustand):
    m, k, b = zustand
    incr = -1 if b else +1
    b_new = 0 if b else 1

    # manuelle Enumeration obsolet
    fzustaende = []

    # ermittle alle möglichen Kombinationen, Kannibalen und Missionare in einem Boot zu plazieren
    for f1 in range(0, max_boat + 1):
        for f2 in range(0, max_boat + 1 - f1):
            if f1 or f2: fzustaende.append([m + incr * f1, k + incr * f2, b_new])

    # Entferne alle, die für m bzw. k kleiner 0 oder größer m_start bzw. k_start sind
    return [[m, k, b] for m, k, b in fzustaende
            if 0 <= k <= k_start and 0 <= m <= m_start]
def is_valid(zustand):
    # is_valid betrachtet nun explizit die Situation auf der rechten sowie linken Seite
    # da bei einer größeren Menge Missionare der vorherige Ausdruck nicht mehr funktioniert
    m_west, k_west, b = zustand
    m_east = m_start - m_west
    k_east = k_start - k_west
    # es gibt im Westen mehr Kannibalen, als Missionare
    if m_west < k_west and m_west > 0: return False
    # es gibt im Osten mehr Kannibalen, als Missionare
    if m_east < k_east and m_east > 0: return False
    return True

def gib_valide_folgezustaende(zustand):
    return [z for z in gib_folgezustaende(zustand) if is_valid(z)]


# Rekursive Suche in die Tiefe (depth-first search with chronological backtracking)
def suche(zustand, history, all_solutions=False, level=0, debug=1):

    if debug: print(level * " ", zustand, " ->", end="")

    if zustand == zielzustand: return (True, history + [zustand])

    fzustaende = gib_valide_folgezustaende(zustand)

    if debug: print("  ", fzustaende)

    if not len(fzustaende): return (False, [])

    for z in fzustaende:
        if z not in history + zustand:
            res1, res2 = suche(z, history + [zustand], all_solutions, level + 1, debug)
            if res1:
                if all_solutions:
                    print("Solution found: ", res1, res2)
                else:
                    return (res1, res2)
        else:
            if debug == 2: print((level + 1) * " " + "repeated", z)
    return (False, [])
#Sucht mit 3 Kannibalen, 3 Missionaren und 2er-Boot
suche(startzustand, [], all_solutions=True, debug=0)

# Sucht mit 4 Kannibalen & Missionaren im 3er-Boot
max_boat = 3
m_start = 4
k_start = 4
startzustand = [m_start, k_start, 1]
suche(startzustand, [], all_solutions=False,debug=2)
# Möglich ist es nun auch, die drei Variablen beliebig zu wählen (nur Vorsicht mir großen Werten, da der Algorithmus einen hohen Aufwand hat)
max_boat = 3
m_start = 3
k_start = 2
startzustand = [m_start, k_start, 1]
suche(startzustand, [], all_solutions=False, debug=2)