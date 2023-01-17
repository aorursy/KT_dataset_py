# Simplistic DBSCAN (vgl. http://www-m9.ma.tum.de/material/felix-klein/clustering/Methoden/Dichteverbundenes_Clustern.php)

# (1) Identifiziere Kern-, Rand- oder Rauschpunkte.
# (2) Lösche alle Rauschpunkte.
# (3) Verbinde Kernpunkte, die gemeinsam in einer ε-Kugel liegen, durch eine Kante.
# (4) Die in einer Komponente des Graphen verbundenen Kernpunkte bilden ein separates Cluster.
# (5) Weise jeden Randpunkt dem Cluster eines benachbarten Kernpunkts zu.