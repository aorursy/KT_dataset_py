from minimization_toolkit import *
%matplotlib inline
f_ = x**2 + y**2
f = sp.lambdify((x,y),f_)

surface(*field(f,-1,1,100),darkgraph(),ax3D())
# random search
def rs(f, n, Bx, By, index=None, cm=np.inf):
    if index is None:
        index = range(n)
    S = np.array(range(n))
    X = np.array([rand(*Bx) for _ in S])
    Y = np.array([rand(*By) for _ in S])
    Z = f(X, Y)
    M = []
    for s in S:
        if Z[s] < cm:
            cm = Z[s]
        M.append(cm)
    return pd.DataFrame({
        'x': X, 'y': Y, 'z': Z,
        'm': M, 'Search': ['Random'] * n }, index=index)
# zooming random search
def zrs(f, n, Bx, By, zooms, Pf=0.8):
    N = [n // zooms] * zooms
    N[-1] += n % zooms
    SS = []
    elapsed = 0
    cm = np.inf
    for i in range(zooms):
        S = rs(f, N[i], Bx, By, index=range(elapsed, elapsed+N[i]), cm=cm)
        A = np.abs(Bx[1]-Bx[0]) * np.abs(By[1]-By[0])
        D = np.sqrt(-A * np.log(1-Pf) / n / np.pi)
        X, Y, M = np.array(S['x']), np.array(S['y']), np.array(S['m'])
        cx, cy, cm = X[-1], Y[-1], M[-1]
        for j in range(N[i])[:0:-1]:
            if not M[j] == M[j-1]:
                cx = X[j]
                cy = Y[j]
                cm = M[j]
                break
        Bx = (cx-D,cx+D)
        By = (cy-D,cy+D)
        SS.append(S)
        elapsed += N[i]
    SS = pd.concat(SS, ignore_index=False)
    SS['Search'] = ['Zooming Random'] * n
    return SS
n=200 ; B=(-1,1) ; z=10 ; Pf=0.8

translucent(*field(f,-1,1,100), darkgraph(), ax3D(), alpha=0.2)
plt.plot(*zdescend(*unbox(rs(f,n,B,B),'x','y','z')),color='red',linewidth=2)
plt.plot(*zdescend(*unbox(zrs(f,n,B,B,z,Pf),'x','y','z')),color='blue',linewidth=2);
N=300 ; n=100 ; B=(-1,1) ; z=2; Pf=0.8

(compare(singles(rs, f, n, B, B),
    singles(zrs, f, n, B, B, z, Pf)) +
p9.scales.scale_color_manual(unbox(COLORS,'blue','orange'))
).draw();

(comparesd(meanssd(rs, N, f, n, B, B),
    meanssd(zrs, N, f, n, B, B, z, Pf))+
p9.scales.scale_color_manual(unbox(COLORS,'blue','orange'))+
p9.scales.ylim(-0.02,0.5)).draw();
h_ = 5*x**2 + y**2
h = sp.lambdify((x,y), h_)
surface(*field(h,-1,1,100), darkgraph(), ax3D())
g = sp.lambdify([x,y], 
    np.array([sp.diff(h_, x),sp.diff(h_, y)]))

gradplot(field(h,-1,1,100),field(g,-1,1,10),
        darkgraph(), ax2D())
sp.vector.gradient(h_)
# fixed gradient descend search
def fgds(f, g, n, D, Bx, By, cm=np.inf):
    p = [rand(*Bx),rand(*Bx)]
    X,Y,Z,G,M = [],[],[],[],[]
    for _ in range(n):
        p = p - (g(*p) / np.linalg.norm(g(*p))) * D
        X.append(p[0])
        Y.append(p[1])
        z = f(*p)
        Z.append(z)
        G.append(g(*p))
        if z<cm: cm=z
        M.append(cm)
    return pd.DataFrame({
        'x':X,'y':Y,'z':Z,'g':G,'m':M,
        'Search':['Fixed Gradient Descend']*n
    })
D=0.1 ; n=30 ; B=(-1,1)

bfield = field(h,-1,1,100)
gfield = field(g,-1,1,10)
path = unbox(fgds(h,g,n,D,B,B),'x','y','z')

travelplot3D(bfield,path,darkgraph(),ax3D())
travelplotgrad(bfield,gfield,path[:-1],darkgraph(),ax2D())
# dinamic gradient descend search
def dgds(f, g, n, D0, Bx, By, cm=np.inf):
    p = [rand(*Bx),rand(*Bx)]
    X,Y,Z,G,M = [],[],[],[],[]
    for s in range(n):
        Ds = D0 *  (1 - s/n)
        p = p - (g(*p) / np.linalg.norm(g(*p))) * Ds
        X.append(p[0])
        Y.append(p[1])
        z = f(*p)
        Z.append(z)
        G.append(g(*p))
        if z<cm: cm=z
        M.append(cm)
    return pd.DataFrame({
        'x':X,'y':Y,'z':Z,'g':G,'m':M,
        'Search':['Dinamic Gradient Descend']*n})
D=0.1 ; n=30 ; B=(-1,1)

bfield = field(h,-1,1,100)
gfield = field(g,-1,1,10)
path = unbox(dgds(h,g,n,D,B,B),'x','y','z')

travelplot3D(bfield,path,darkgraph(),ax3D())
travelplotgrad(bfield,gfield,path[:-1],darkgraph(),ax2D())
N=200 ; n=30 ; B=(-1,1) ; D = 0.1
(comparesd(
    meanssd(rs, N, h, n, B, B),
    meanssd(zrs, N, h, n, B, B, zooms=2, Pf=0.8),
    meanssd(fgds, N, h, g, n, D, B, B),
    meanssd(dgds, N, h, g, n, D, B, B),
    res=3) +
p9.scales.scale_color_manual(unbox(COLORS,'red','green','blue','orange')) +
p9.scales.ylim(-0.1,1.3)).draw();

(comparesd(
    meanssd(fgds, N, h, g, n, D, B, B),
    meanssd(dgds, N, h, g, n, D, B, B),
    res=8) +
 p9.scales.scale_color_manual(unbox(COLORS,'red','green')) +
 p9.scales.ylim(-0.02,0.041) +
 p9.scales.xlim(12.5,30)).draw();
psi_ = -sp.exp(-((x-1)**2+(y-1)**2)/0.3**2)-1.5*sp.exp(-16*((x-2)**2+(y-2)**2))
psi  = sp.lambdify((x,y),psi_)
translucent(*field(psi,-1,5,300), darkgraph(), ax3D())
gpsi = sp.lambdify([x,y], np.array([sp.diff(psi_, x),sp.diff(psi_, y)]))
sp.vector.gradient(psi_)
# simulated annealing search
def sas(f, g, n, D0, Bx, By, T0, cm=np.inf, d=3):
    T = np.linspace(T0, 0, n)[::-1]
    d = np.array([(rand(-d,d),rand(-d,d)) for _ in T])
    p = [rand(*Bx),rand(*Bx)]
    X, Y, Z, G, M = [], [], [], [], []
    for s in range(n):
        Ds = D0 *  (1 - s/n)
        grad = g(*p) + d[s]
        p1 = p - ((grad / np.linalg.norm(grad)) * Ds)
        f0 = f(*p)
        f1 = f(*p1)
        if f1 < f0 or rand(0,1) > np.exp((f0-f1)*T[s]) :
            p = p1
        X.append(p[0]); Y.append(p[1])
        z = f(*p)
        Z.append(z); G.append(g(*p))
        if z < cm:
            cm = z
        M.append(cm)
    return pd.DataFrame({
        'x':X,'y':Y,'z':Z,'g':G,'m':M,
        'Search':['Simulated Anealing']*n
    })
D0=0.6 ; n=150 ; B=(-1,5) ; T0=1000

translucent(*field(psi,-1,5,500), darkgraph(), ax3D(), alpha=0.2)
for color in ['red','green','blue', 'yellow']:
    plt.plot(*unbox(sas(psi,gpsi,n,D0,B,B,T0),'x','y','z'),
            color=color,linewidth=2)
N=100 ; n=150 ; B=(-1,5) ; D=0.6
z=2 ; Pf=0.8 ; T0=100

(compare(
    means(zrs,N,psi,n,B,B,z,Pf),
    means(fgds,N,psi,gpsi,n,D,B,B),
    means(dgds,N,psi,gpsi,n,D,B,B),
    means(sas,N,psi,gpsi,n,D,B,B,T0)) +
p9.scales.scale_color_manual(unbox(COLORS,'green','orange','red','blue'))
).draw();
N=100 ; n=150 ; B=(-1,5) ; D=0.6
z=4 ; Pf=0.8 ; T0=100

(comparesd(
    meanssd(dgds,N,psi,gpsi,n,D,B,B), 
    meanssd(sas,N,psi,gpsi,n,D,B,B,T0))+
 p9.scales.scale_color_manual(unbox(COLORS,'green','red'))
).draw();
N=100 ; n=2000 ; B=(-1,5) ; D=0.6
z=2 ; Pf=0.8 ; T0=100 ; d=3

dfzrs = meanssd(zrs,N,psi,n,B,B,z,Pf)
dfdgds = meanssd(dgds,N,psi,gpsi,n,D,B,B)
dfsas = meanssd(sas,N,psi,gpsi,n,D,B,B,T0,d)
(comparesd(dfsas,dfdgds, dfzrs, res=4) +
p9.scales.scale_color_manual(unbox(COLORS,'green','red','orange'))).draw();
N=50 ; n=10**4 ; B=(-1,5)
z=2 ; Pf=0.8

(comparesd(
    meanssd(rs,N,psi,n,B,B),
    meanssd(zrs,N,psi,n,B,B,z,Pf))  +
p9.scales.scale_color_manual(unbox(COLORS,'blue','orange'))+
p9.scales.ylim(-1.6,-1)).draw();