Musica = ["Clássica", "Rock", "Jazz", "Eletrônica", "Pop", "Hip Hop"]
print(len(Musica))

print(Musica.index("Clássica"))

print(Musica[4])

print(Musica[-4])

print(Musica[:3])

print(Musica[3:])

print(Musica[2:4])

print(Musica[-4:-1])
Artista = list(("Beethoven", "Kiss", "Mile Davis", "Skrillex", "Lady Gaga", "50cent"))
Musica.append("MPB")
print("Musica.append('MPB') =", Musica)
Artista.insert(6, "Tim Maia")
print("Artista.insert(6, 'Tim Maia') =", Artista)
Musica.remove("Pop")
print("Musica.remove('Pop') =", Musica)
Artista.pop(4)
print("Artista.pop(4) =", Artista)
Artista[1] = "Metallica"
print("Artista[index 1] = 'Metallica' =", Artista)
Musica.reverse()
print("Musica.reverse() =", Musica)
backup_musica = Musica.copy()
print("backup_musica = Musica.copy() =", backup_musica)
Musica.sort()
print("Musica.sort() =", Musica)

Musica.sort(reverse=True)
print("\nMusica.sort(reverse=True) =", Musica)

def tamanho(x):
    return len(x)

Musica.sort(key=tamanho)

print("\nMusica.sort(key=tamanho) =", Musica)
Musica.extend(backup_musica)

print("Musica.extend(backup_musica) =", Musica)
conta = Musica.count("Rock")
print("conta = Musica.count('Rock') =", conta)
Musica.clear()

print("Musica.clear() =", Musica)