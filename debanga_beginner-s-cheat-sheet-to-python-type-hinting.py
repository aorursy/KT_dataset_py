def add_two_integers(x: int = 1, y: int = 2) -> int:
    return x + y
length: int # no value at runtime until assigned
length: int = 10

is_square: bool # no value at runtime until assigned
is_square: bool = False
    
width: float # no value at runtime until assigned
width: float = 100
    
name: str # no value at runtime until assigned
name: str = "rectangle"
from typing import List, Set, Dict, Tuple, Optional
x: List[int] = [1, 2]
x: Set[str] = {'rect', 'square'}
x: Dict[str, float] = {'length': 10.0, 'width': 100.0}
x: Tuple[str, float, float] = ("rect", 10.0, 100.0)
    
x: Tuple[int, ...] = (1, 2, 3) # Variable size tuple
def compare_numbers(x: int) -> int:
    if x<10:
        return 1
    elif x>10:
        return 0
    else:
        return None
    

x: Optional[int] = compare_numbers(10)
from typing import Callable, Iterator, Union
x: Callable[[int, int], int] = add_two_integers
def generator(n: int) -> Iterator[int]:
    i = 0
    while i < n:
        yield i
        i += 1
def add_two_integers_or_floats(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
    return x + y
F = List[float] # Aliasing list of floats

a: F = [1.0, 4.6, 9.1]
from typing import TypeVar, Iterable, DefaultDict

Relation = Tuple[T, T]
def create_tree(tuples: Iterable[Relation]) -> DefaultDict[T, List[T]]:
    tree: DefaultDict[T, List[T]] = defaultdict(list)
    for idx, child in enumerate(tuples):
        tree[idx].append(child)

    return tree

print(create_tree([(2.0,1.0), (3.0,1.0), (4.0,3.0), (1.0,6.0)]))