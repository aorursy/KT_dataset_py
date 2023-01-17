import matplotlib.pyplot as plt

import numpy as np
def game(p, verbose=False):

    q=1-p

    tiebreak=p**2/(p**2+q**2)

    winpct=p**4+(4*p**4*q)+(10*p**4*q**2)+(20*p**3*q**3*tiebreak)

    if verbose:print(f"p={p}, q={q}, tiebreak={tiebreak}, win pct={winpct}")

    return winpct



def set(p, verbose=False):

    q=1-p

    tiebreak=p**2/(p**2+q**2)

    winpct=p**6+(6*p**6*q)+(21*p**6*q**2)+(56*p**6*q**3)+(126*p**6*q**4)+(252*p**5*q**5*tiebreak)

    if verbose:print(f"p={p}, q={q}, tiebreak={tiebreak}, win pct={winpct}")

    return winpct



def womensmatch(p, verbose=False):

    q=1-p

    winpct=p**2+(2*p**2*q)

    if verbose:print(f"p={p}, q={q}, win pct={winpct}")

    return winpct



def mensmatch(p, verbose=False):

    q=1-p

    winpct=p**3+(3*p**3*q)+(6*p**3*q**2)

    if verbose:print(f"p={p}, q={q}, win pct={winpct}")

    return winpct

    

    

    
gamepct=game(0.55,True)

setpct=set(gamepct,True)

wmpct=womensmatch(setpct,True)

mmpct=mensmatch(setpct,True)

wtourneypct=wmpct**7

mtourneypct=mmpct**7

print(f"55% win percentage for each point yields {gamepct*100}% win percentage for each game and {setpct*100}% win percentage for each set.")

print(f"Which yields a {wmpct*100}% win percentage in a women's match to 2 and a {mmpct*100}% win percentage in a men's match to 3.")

print(f"Your chances of doing that seven times in a row are {wtourneypct*100}% in a women's tournament and {mtourneypct*100}% in a men's tournament")
p=np.arange(0,1,0.01)

plt.plot(p, game(p))

plt.plot(p, set(game(p)))

plt.plot(p, womensmatch(set(game(p))))

plt.plot(p, mensmatch(set(game(p))))

plt.plot(p, womensmatch(set(game(p)))**7)

plt.plot(p, mensmatch(set(game(p)))**7)

plt.xlabel("Win percentage for each point")

plt.xlim(0,1)

plt.legend(['game %', 'set %', "women's match%", "men's match%", "women's tournament%", "men's tournament%"], loc='upper left')

plt.rcParams['figure.figsize'] = [10, 10]

plt.show()