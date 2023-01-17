import sqlite3

try:
    con = sqlite3.connect('test.db')
    cur = con.cursor()

    cur.executescript("DROP TABLE IF EXISTS Pets;")

    cur.executescript("CREATE TABLE Pets(Id INT, Name TEXT, Price INT);")

    cur.executescript("INSERT INTO Pets VALUES(1, 'Cat', 400);")
    cur.executescript("INSERT INTO Pets VALUES(2, 'Dog', 600);")

    pets = ((3, 'Rabbit', 200),
            (4, 'Bird', 60),
            (5, 'Goat', 500))

    cur.executemany("INSERT INTO Pets VALUES(?, ?, ?)", pets)

    # for p in pets:
    #     cur.executescript(f"INSERT INTO Pets VALUES({p[0]}, '{p[1]}', {p[2]});")

    con.commit()

    cur.execute("SELECT * FROM Pets")

    data = cur.fetchall()

    output = []

    for row in data:
        print(row)
        output.append(f"ID={row[0]}, NAME={row[1]}, PRICE={row[2]}.")

    print("<br>\n".join(output))

except sqlite3.Error as e:
    if con:
        con.rollback()
finally:
    if con:
        con.close()

